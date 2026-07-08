import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

# pyrefly: ignore [missing-import]
from src.config import settings
# pyrefly: ignore [missing-import]
from src.ingestion.chunker import DocumentChunk

class GroundedGenerator:
    """Generates answers that are grounded in search results with inline citations."""
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self._openai_client = openai_client

    @property
    def openai_client(self) -> OpenAI:
        if not self._openai_client:
            self._openai_client = OpenAI(api_key=settings.openai_api_key or "mock_key")
        return self._openai_client

    def format_context_blocks(self, chunks: List[DocumentChunk]) -> str:
        """Formats a list of DocumentChunks into numbered context blocks for the prompt."""
        if not chunks:
            return "(No context blocks available)"
            
        blocks = []
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.source_file
            section = f" (Section: {chunk.section_heading})" if chunk.section_heading else ""
            page = f" (Page: {chunk.page_number})" if chunk.page_number is not None else ""
            
            block = (
                f"Context Block [{idx}]\n"
                f"Source: {source}{section}{page}\n"
                f"Content: {chunk.text.strip()}"
            )
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    def get_system_prompt(self) -> str:
        """Returns the system instructions for the LLM to enforce grounded generation and citation."""
        return (
            "You are a highly precise and grounded question-answering assistant. Your task is to answer "
            "the user's question ONLY using the facts directly mentioned in the provided Context Blocks.\n\n"
            "Strict Rules for Your Response:\n"
            "1. Grounding: Answer the question using ONLY the facts directly mentioned in the provided Context Blocks. "
            "Do not assume, extrapolate, or bring in any outside knowledge. Every claim you make must be fully "
            "supported by the context.\n"
            "2. Citation: Every claim or fact you state MUST be followed by the appropriate bracketed citation corresponding "
            "to the Context Block number (e.g., [1], [2], [1][3]). Do not combine citations into ranges (use [1][2] instead of [1-2]). "
            "The citation must point to the specific Context Block(s) where the fact was found.\n"
            "3. Unanswerable Questions: If the provided Context Blocks do not contain enough information to fully answer the question, "
            "or if you cannot answer the question based strictly on the provided context, you must state exactly and only: "
            "\"I do not have enough information in the provided context to answer this question.\"\n"
            "4. Format: Be concise, clear, and professional. Ensure citations are placed immediately after the relevant sentence or clause they support."
        )

    def get_fallback_system_prompt(self) -> str:
        """Returns the fallback system instructions for generating a structured search report when retrieval confidence is low."""
        return (
            "You are a highly helpful and transparent documentation search assistant. The search system retrieved some "
            "documents for the user's question, but the retrieval confidence is below the threshold, meaning the context "
            "is likely insufficient or only partially relevant.\n\n"
            "Do not attempt to answer the question using external knowledge or make up answers. Instead, analyze the "
            "provided Context Blocks and the user's question, and output a structured report explaining the search results:\n\n"
            "1. What We Found: Summarize what relevant details or partial information (if any) WERE found in the context blocks related to the question.\n"
            "2. What Is Missing: Clearly explain what key information or concepts required to fully answer the question were NOT found in the context blocks.\n"
            "3. Recommended Documents: List the source file names and section headings of the retrieved documents that were searched, suggesting which ones the user might want to check manually.\n\n"
            "Format your response in clean Markdown with clear headings: '### What We Found', '### What Is Missing', and '### Recommended Documents'."
        )

    def get_user_prompt(self, question: str, formatted_context: str) -> str:
        """Combines the question and formatted context into a user prompt."""
        return (
            f"Context Blocks:\n"
            f"======================\n"
            f"{formatted_context}\n"
            f"======================\n\n"
            f"Question: {question}\n"
        )

    def _parse_claims_and_citations(self, text: str) -> List[Dict[str, Any]]:
        """Parses individual sentences and associates them with their cited source indices."""
        # Simple regex split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        results = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Find all bracketed citations like [1], [2]
            citation_indices = re.findall(r'\[(\d+)\]', sentence)
            if citation_indices:
                # Remove the bracketed citations from the sentence to get a clean claim
                claim = re.sub(r'\[\d+\]', '', sentence).strip()
                # Clean up multiple spaces, trailing punctuation if any, and formatting
                claim = re.sub(r'\s+', ' ', claim)
                claim = re.sub(r'\s+([.,!?])', r'\1', claim)
                for index_str in citation_indices:
                    results.append({
                        "claim": claim,
                        "citation_index": int(index_str),
                        "original_sentence": sentence
                    })
        return results

    def _judge_claim(self, claim: str, context: str) -> bool:
        """Uses OpenAI chat completions as a judge to verify if context supports the claim."""
        system_prompt = (
            "You are an expert fact-checking assistant. Your task is to verify if a given claim "
            "is directly and fully supported by the provided context block.\n\n"
            "Instructions:\n"
            "1. Verify if the claim contains only facts that are directly stated or directly implied by the context.\n"
            "2. If the claim is fully supported by the context, output exactly: VERIFIED\n"
            "3. If the claim is not supported, partially supported, contradicted, or if the context does "
            "not contain enough information, output exactly: UNSUPPORTED\n"
            "4. Do NOT include any other text, reasoning, or explanation. Output only VERIFIED or UNSUPPORTED."
        )
        
        user_prompt = (
            f"Context Block:\n"
            f"\"\"\"\n"
            f"{context.strip()}\n"
            f"\"\"\"\n\n"
            f"Claim to Verify:\n"
            f"\"\"\"\n"
            f"{claim.strip()}\n"
            f"\"\"\"\n"
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=5
            )
            decision = (response.choices[0].message.content or "").strip().upper()
            return "VERIFIED" in decision
        except Exception as e:
            print(f"Error during claim verification call: {e}")
            return False

    def verify_citations(self, answer: str, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """Parses citations from the answer and verifies each claim against the cited context chunk."""
        parsed_claims = self._parse_claims_and_citations(answer)
        verified_results = []
        
        for item in parsed_claims:
            idx = item["citation_index"]
            claim = item["claim"]
            
            # Retrieve the cited chunk (1-based index)
            if idx <= 0 or idx > len(chunks):
                verified_results.append({
                    "claim": claim,
                    "citation_index": idx,
                    "source_file": "Unknown",
                    "result": "UNSUPPORTED"
                })
                continue
                
            cited_chunk = chunks[idx - 1]
            is_supported = self._judge_claim(claim, cited_chunk.text)
            
            verified_results.append({
                "claim": claim,
                "citation_index": idx,
                "source_file": cited_chunk.source_file,
                "result": "VERIFIED" if is_supported else "UNSUPPORTED"
            })
            
        return verified_results

    def calculate_citation_coverage(self, answer: str, verification_results: List[Dict[str, Any]]) -> float:
        """Calculates what percentage of sentences/claims in the answer have verified citations."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 1.0
            
        # Group verification results by their claim string
        sentence_status = {}
        for res in verification_results:
            orig = res.get("claim", "")
            verdict = res.get("result", "UNSUPPORTED")
            
            if orig not in sentence_status:
                sentence_status[orig] = True
            if verdict == "UNSUPPORTED":
                sentence_status[orig] = False
                
        # Count sentences that are fully verified
        verified_count = 0
        for sentence in sentences:
            clean_s = re.sub(r'\[\d+\]', '', sentence).strip()
            clean_s = re.sub(r'\s+', ' ', clean_s)
            clean_s = re.sub(r'\s+([.,!?])', r'\1', clean_s)
            
            if clean_s in sentence_status and sentence_status[clean_s] is True:
                verified_count += 1
                
        return float(verified_count / len(sentences))

    def _evaluate_completeness(self, question: str, answer: str) -> float:
        """Uses OpenAI chat completions as a judge to score answer completeness from 0.0 to 1.0."""
        system_prompt = (
            "You are an expert quality assurance assistant. Your task is to evaluate if a generated answer "
            "fully addresses all parts of a given question.\n\n"
            "Instructions:\n"
            "1. Output a single completeness score between 0.0 and 1.0 (inclusive).\n"
            "2. 1.0 means the answer completely and fully addresses all parts of the question (or correctly "
            "states that there is not enough information under the system rules).\n"
            "3. 0.0 means the answer does not address the question at all or is completely irrelevant.\n"
            "4. Output ONLY the numeric float score (e.g., 1.0, 0.75, 0.5, 0.0). Do NOT include any other text."
        )
        
        user_prompt = (
            f"Question:\n"
            f"\"\"\"\n"
            f"{question.strip()}\n"
            f"\"\"\"\n\n"
            f"Generated Answer:\n"
            f"\"\"\"\n"
            f"{answer.strip()}\n"
            f"\"\"\"\n"
        )
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=5
            )
            score_str = (response.choices[0].message.content or "").strip()
            match = re.search(r"\d+(?:\.\d+)?", score_str)
            if match:
                score = float(match.group(0))
                return min(max(score, 0.0), 1.0)
            return 0.0
        except Exception as e:
            print(f"Error during completeness check: {e}")
            return 0.0

    def calculate_confidence_score(
        self,
        retrieval_score: float,
        citation_coverage: float,
        completeness_score: float,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Calculates a composite confidence score based on retrieval, citation, and completeness."""
        w = weights or {
            "retrieval": 0.33,
            "citation": 0.33,
            "completeness": 0.34
        }
        composite = (
            w.get("retrieval", 0.33) * retrieval_score +
            w.get("citation", 0.33) * citation_coverage +
            w.get("completeness", 0.34) * completeness_score
        )
        return {
            "retrieval_confidence": retrieval_score,
            "citation_coverage": citation_coverage,
            "answer_completeness": completeness_score,
            "composite_score": round(composite, 4)
        }

    def generate_response(
        self,
        question: str,
        chunks_or_results: List[Any],
        weights: Optional[Dict[str, float]] = None,
        retrieval_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Generates a grounded response with citations and calculates a confidence score report."""
        
        # Normalize inputs
        chunks: List[DocumentChunk] = []
        scores: List[float] = []
        for item in chunks_or_results:
            if isinstance(item, dict) and "chunk" in item:
                chunks.append(item["chunk"])
                scores.append(item.get("score", 1.0))
            elif isinstance(item, DocumentChunk):
                chunks.append(item)
                scores.append(1.0)
                
        formatted_context = self.format_context_blocks(chunks)
        user_prompt = self.get_user_prompt(question, formatted_context)
        
        if not chunks:
            system_prompt = self.get_system_prompt()
            answer = "I do not have enough information in the provided context to answer this question."
            confidence = self.calculate_confidence_score(0.0, 1.0, 1.0, weights)
            return {
                "answer": answer,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "verification_results": [],
                "confidence_report": confidence,
                "fallback_triggered": True
            }
            
        # Determine if retrieval score is below threshold
        max_retrieval_score = max(scores) if scores else 0.0
        is_low_confidence = max_retrieval_score < retrieval_threshold
        
        if is_low_confidence:
            system_prompt = self.get_fallback_system_prompt()
        else:
            system_prompt = self.get_system_prompt()
            
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content or ""
            answer = answer.strip()
            
            if is_low_confidence:
                # If fallback was triggered, we did not produce inline citations
                verification_results = []
                citation_coverage = 1.0  # structured metadata report doesn't make unsupported factual claims
                
                # Retrieve completeness score
                completeness_score = self._evaluate_completeness(question, answer)
                
                confidence = self.calculate_confidence_score(
                    max_retrieval_score,
                    citation_coverage,
                    completeness_score,
                    weights
                )
                
                return {
                    "answer": answer,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "verification_results": [],
                    "confidence_report": confidence,
                    "fallback_triggered": True
                }
            else:
                verification_results = self.verify_citations(answer, chunks)
                
                # 1. Retrieval confidence (based on cited chunks)
                cited_indices = {res["citation_index"] for res in verification_results}
                valid_cited_indices = {idx for idx in cited_indices if 1 <= idx <= len(scores)}
                if valid_cited_indices:
                    retrieval_score = sum(scores[idx - 1] for idx in valid_cited_indices) / len(valid_cited_indices)
                else:
                    retrieval_score = scores[0] if scores else 0.0
                    
                # 2. Citation coverage
                citation_coverage = self.calculate_citation_coverage(answer, verification_results)
                
                # 3. Completeness
                completeness_score = self._evaluate_completeness(question, answer)
                
                # 4. Composite confidence score
                confidence = self.calculate_confidence_score(
                    retrieval_score,
                    citation_coverage,
                    completeness_score,
                    weights
                )
                
                return {
                    "answer": answer,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "verification_results": verification_results,
                    "confidence_report": confidence,
                    "fallback_triggered": False
                }
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise e
