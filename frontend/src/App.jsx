import { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  Layers, 
  FileText, 
  Upload, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  Clock, 
  Shield, 
  Database,
  Cpu
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  
  // Pipeline outputs
  const [hybridResult, setHybridResult] = useState(null);
  const [denseResult, setDenseResult] = useState(null);
  
  // Document listings
  const [documents, setDocuments] = useState([]);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [notification, setNotification] = useState(null);
  
  // Interactive citation highlights
  const [highlightedHybridIdx, setHighlightedHybridIdx] = useState(null);
  const [highlightedDenseIdx, setHighlightedDenseIdx] = useState(null);
  
  // Refs for scrolling to citations
  const hybridChunkRefs = useRef({});
  const denseChunkRefs = useRef({});
  
  const fileInputRef = useRef(null);

  // Fetch indexed documents list on load
  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE}/v1/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const triggerNotification = (message) => {
    setNotification(message);
    setTimeout(() => {
      setNotification(null);
    }, 4000);
  };

  // Handle RAG pipeline query execution
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setHybridResult(null);
    setDenseResult(null);
    setHighlightedHybridIdx(null);
    setHighlightedDenseIdx(null);
    
    try {
      // Build both fetch promises: Hybrid (0.5/0.5 weights) vs. Dense-only (1.0/0.0 weights)
      const hybridPromise = fetch(`${API_BASE}/v1/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          threshold: 0.7,
          dense_weight: 0.5,
          sparse_weight: 0.5,
          k: 5
        })
      });

      const densePromise = fetch(`${API_BASE}/v1/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          threshold: 0.7,
          dense_weight: 1.0,
          sparse_weight: 0.0,
          k: 5
        })
      });

      // Execute in parallel
      const [hybridRes, denseRes] = await Promise.all([hybridPromise, densePromise]);
      
      if (hybridRes.ok) {
        const data = await hybridRes.json();
        setHybridResult(data);
      }
      
      if (denseRes.ok) {
        const data = await denseRes.json();
        setDenseResult(data);
      }
    } catch (err) {
      console.error('Error executing query:', err);
      triggerNotification('Failed to retrieve query response.');
    } finally {
      setLoading(false);
    }
  };

  // Highlight and scroll to chunk when citation badge is clicked
  const handleCitationClick = (idx, isHybrid = true) => {
    if (isHybrid) {
      setHighlightedHybridIdx(idx);
      const element = hybridChunkRefs.current[idx];
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      setTimeout(() => setHighlightedHybridIdx(null), 3000);
    } else {
      setHighlightedDenseIdx(idx);
      const element = denseChunkRefs.current[idx];
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      setTimeout(() => setHighlightedDenseIdx(null), 3000);
    }
  };

  // File Upload Ingestion handler
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_BASE}/v1/ingest?chunking_strategy=recursive`, {
        method: 'POST',
        body: formData
      });
      
      if (res.ok) {
        triggerNotification(`Successfully ingested and indexed ${file.name}!`);
        fetchDocuments();
      } else {
        const data = await res.json();
        triggerNotification(`Failed to ingest file: ${data.detail || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error uploading file:', err);
      triggerNotification('Connection error during file ingestion.');
    } finally {
      setUploadLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Custom Markdown & Citation Parser
  const parseMarkdownAndCitations = (text, isHybrid = true) => {
    if (!text) return null;
    
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      let cleanLine = line;
      
      // Bold syntax
      cleanLine = cleanLine.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      // Inline code
      cleanLine = cleanLine.replace(/`(.*?)`/g, '<code>$1</code>');
      
      const citationRegex = /\[(\d+)\]/g;
      const parts = [];
      let lastIndex = 0;
      let match;
      
      while ((match = citationRegex.exec(cleanLine)) !== null) {
        const matchIndex = match.index;
        const citationNumber = match[1];
        
        if (matchIndex > lastIndex) {
          parts.push(
            <span 
              key={`text-${lastIndex}`} 
              dangerouslySetInnerHTML={{ __html: cleanLine.substring(lastIndex, matchIndex) }} 
            />
          );
        }
        
        parts.push(
          <span 
            key={`badge-${matchIndex}`} 
            className="citation-badge"
            onClick={() => handleCitationClick(parseInt(citationNumber), isHybrid)}
          >
            [{citationNumber}]
          </span>
        );
        
        lastIndex = citationRegex.lastIndex;
      }
      
      if (lastIndex < cleanLine.length) {
        parts.push(
          <span 
            key={`text-end`} 
            dangerouslySetInnerHTML={{ __html: cleanLine.substring(lastIndex) }} 
          />
        );
      }
      
      if (line.startsWith('# ')) {
        return <h1 key={idx}>{parts}</h1>;
      } else if (line.startsWith('## ')) {
        return <h2 key={idx}>{parts}</h2>;
      } else if (line.startsWith('### ')) {
        return <h3 key={idx}>{parts}</h3>;
      } else if (line.startsWith('- ') || line.startsWith('* ')) {
        return <li key={idx} style={{ marginLeft: '1rem' }}>{parts}</li>;
      }
      
      return <p key={idx}>{parts}</p>;
    });
  };

  return (
    <div className="dashboard-container">
      {/* Toast Notification */}
      {notification && (
        <div className="notification">
          <CheckCircle size={18} />
          <span>{notification}</span>
        </div>
      )}

      {/* Header */}
      <header className="dashboard-header">
        <div className="logo-section">
          <h1>Hybrid Search RAG</h1>
          <p>Clickable Citations, Multidimensional Scoring, and Side-by-Side Ingestion comparisons.</p>
        </div>
        
        <div className="header-controls">
          <div className="toggle-container" onClick={() => setCompareMode(!compareMode)}>
            <Layers size={18} className={compareMode ? 'text-purple-400' : 'text-gray-400'} />
            <span className={`toggle-label ${compareMode ? 'active' : ''}`}>
              Compare vs. Dense-Only
            </span>
            <div className={`switch-box ${compareMode ? 'active' : ''}`}>
              <div className="switch-handle" />
            </div>
          </div>
        </div>
      </header>

      {/* Search Input Bar */}
      <div className="search-area">
        <form onSubmit={handleSearch} className="search-box">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSearch(e);
              }
            }}
            placeholder="Ask a question about auth configurations, database specs, deployment pipelines, or metrics..."
            disabled={loading}
          />
          <button type="submit" className="search-button" disabled={loading || !query.trim()}>
            <Search size={16} />
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {/* Main Workspace Panels */}
      <div className={`workspace-grid ${compareMode ? 'comparison' : ''}`}>
        
        {/* HYBRID ROUTE COLUMN */}
        <div className="pipeline-column">
          {compareMode && (
            <div className="column-header-tag hybrid">
              <Cpu size={16} />
              Hybrid Fusion Pipeline (Production)
            </div>
          )}

          {loading && !hybridResult && (
            <div className="glass-panel">
              <div className="skeleton skeleton-text" style={{ width: '40%' }} />
              <div className="skeleton skeleton-text" />
              <div className="skeleton skeleton-text" />
              <div className="skeleton skeleton-text short" />
            </div>
          )}

          {hybridResult && (
            <>
              {/* Answer Panel */}
              <div className="glass-panel">
                <h3 className="panel-title">
                  <Cpu size={18} className="text-purple-400" />
                  Grounded Response
                </h3>
                
                {hybridResult.fallback_triggered && (
                  <div className="fallback-alert">
                    <AlertTriangle size={20} className="text-rose-500" />
                    <div className="fallback-alert-text">
                      <h4>Fallback Report Triggered (Low Confidence)</h4>
                      <p>Retrieved context score was below 0.7. Switched to transparent search mapping.</p>
                    </div>
                  </div>
                )}
                
                <div className="answer-content">
                  {parseMarkdownAndCitations(hybridResult.answer, true)}
                </div>
              </div>

              {/* Confidence Metrics Panel */}
              <div className="glass-panel">
                <h3 className="panel-title">
                  <Shield size={18} className="text-teal-400" />
                  Pipeline Confidence Breakdown
                </h3>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-card-header">
                      <span className="metric-name">Composite Score</span>
                      <Info size={14} className="text-gray-400" />
                    </div>
                    <span className="metric-value">
                      {(hybridResult.confidence_report.composite_score * 100).toFixed(0)}%
                    </span>
                    <div className="metric-progress-bg">
                      <div 
                        className="metric-progress-fill composite" 
                        style={{ width: `${hybridResult.confidence_report.composite_score * 100}%` }} 
                      />
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-card-header">
                      <span className="metric-name">Retrieval Score</span>
                    </div>
                    <span className="metric-value">
                      {(hybridResult.confidence_report.retrieval_confidence * 100).toFixed(0)}%
                    </span>
                    <div className="metric-progress-bg">
                      <div 
                        className="metric-progress-fill retrieval" 
                        style={{ width: `${hybridResult.confidence_report.retrieval_confidence * 100}%` }} 
                      />
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-card-header">
                      <span className="metric-name">Citation Coverage</span>
                    </div>
                    <span className="metric-value">
                      {(hybridResult.confidence_report.citation_coverage * 100).toFixed(0)}%
                    </span>
                    <div className="metric-progress-bg">
                      <div 
                        className="metric-progress-fill coverage" 
                        style={{ width: `${hybridResult.confidence_report.citation_coverage * 100}%` }} 
                      />
                    </div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-card-header">
                      <span className="metric-name">Completeness Score</span>
                    </div>
                    <span className="metric-value">
                      {(hybridResult.confidence_report.completeness_score * 100).toFixed(0)}%
                    </span>
                    <div className="metric-progress-bg">
                      <div 
                        className="metric-progress-fill completeness" 
                        style={{ width: `${hybridResult.confidence_report.completeness_score * 100}%` }} 
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Retrieved Sources Inside Column in comparison mode */}
              {compareMode && (
                <div className="glass-panel">
                  <h3 className="panel-title">
                    <Database size={18} className="text-purple-400" />
                    Retrieved Context Chunks (Ranked)
                  </h3>
                  <div className="sources-list">
                    {hybridResult.retrieved_chunks.map((chunk, idx) => (
                      <div 
                        key={idx} 
                        ref={el => hybridChunkRefs.current[idx + 1] = el}
                        className={`source-item ${highlightedHybridIdx === idx + 1 ? 'highlighted' : ''}`}
                      >
                        <div className="source-item-header">
                          <div className="source-title">
                            <span className="citation-badge" style={{ cursor: 'default' }}>
                              [{idx + 1}]
                            </span>
                            <span>{chunk.source_file}</span>
                            {chunk.section_heading && (
                              <span className="meta-badge">{chunk.section_heading}</span>
                            )}
                          </div>
                          <span className="source-score">
                            Score: {chunk.score.toFixed(4)}
                          </span>
                        </div>
                        <p className="source-snippet">{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* DENSE-ONLY ROUTE COLUMN (When Comparison Mode is active) */}
        {compareMode && (
          <div className="pipeline-column">
            <div className="column-header-tag dense">
              <Database size={16} />
              Dense-Only Retrieval Pipeline
            </div>

            {loading && !denseResult && (
              <div className="glass-panel">
                <div className="skeleton skeleton-text" style={{ width: '40%' }} />
                <div className="skeleton skeleton-text" />
                <div className="skeleton skeleton-text" />
                <div className="skeleton skeleton-text short" />
              </div>
            )}

            {denseResult && (
              <>
                {/* Answer Panel */}
                <div className="glass-panel">
                  <h3 className="panel-title">
                    <Database size={18} className="text-blue-400" />
                    Dense-Only Response
                  </h3>
                  
                  {denseResult.fallback_triggered && (
                    <div className="fallback-alert">
                      <AlertTriangle size={20} className="text-rose-500" />
                      <div className="fallback-alert-text">
                        <h4>Fallback Report Triggered (Low Confidence)</h4>
                        <p>Retrieved context score was below 0.7.</p>
                      </div>
                    </div>
                  )}
                  
                  <div className="answer-content">
                    {parseMarkdownAndCitations(denseResult.answer, false)}
                  </div>
                </div>

                {/* Metrics Breakdown */}
                <div className="glass-panel">
                  <h3 className="panel-title">
                    <Shield size={18} className="text-teal-400" />
                    Pipeline Confidence Breakdown
                  </h3>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <div className="metric-card-header">
                        <span className="metric-name">Composite Score</span>
                      </div>
                      <span className="metric-value">
                        {(denseResult.confidence_report.composite_score * 100).toFixed(0)}%
                      </span>
                      <div className="metric-progress-bg">
                        <div 
                          className="metric-progress-fill composite" 
                          style={{ width: `${denseResult.confidence_report.composite_score * 100}%` }} 
                        />
                      </div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-card-header">
                        <span className="metric-name">Retrieval Score</span>
                      </div>
                      <span className="metric-value">
                        {(denseResult.confidence_report.retrieval_confidence * 100).toFixed(0)}%
                      </span>
                      <div className="metric-progress-bg">
                        <div 
                          className="metric-progress-fill retrieval" 
                          style={{ width: `${denseResult.confidence_report.retrieval_confidence * 100}%` }} 
                        />
                      </div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-card-header">
                        <span className="metric-name">Citation Coverage</span>
                      </div>
                      <span className="metric-value">
                        {(denseResult.confidence_report.citation_coverage * 100).toFixed(0)}%
                      </span>
                      <div className="metric-progress-bg">
                        <div 
                          className="metric-progress-fill coverage" 
                          style={{ width: `${denseResult.confidence_report.citation_coverage * 100}%` }} 
                        />
                      </div>
                    </div>

                    <div className="metric-card">
                      <div className="metric-card-header">
                        <span className="metric-name">Completeness Score</span>
                      </div>
                      <span className="metric-value">
                        {(denseResult.confidence_report.completeness_score * 100).toFixed(0)}%
                      </span>
                      <div className="metric-progress-bg">
                        <div 
                          className="metric-progress-fill completeness" 
                          style={{ width: `${denseResult.confidence_report.completeness_score * 100}%` }} 
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Retrieved Sources Inside Column in comparison mode */}
                <div className="glass-panel">
                  <h3 className="panel-title">
                    <Database size={18} className="text-blue-400" />
                    Retrieved Context Chunks (Ranked)
                  </h3>
                  <div className="sources-list">
                    {denseResult.retrieved_chunks.map((chunk, idx) => (
                      <div 
                        key={idx} 
                        ref={el => denseChunkRefs.current[idx + 1] = el}
                        className={`source-item ${highlightedDenseIdx === idx + 1 ? 'highlighted' : ''}`}
                      >
                        <div className="source-item-header">
                          <div className="source-title">
                            <span className="citation-badge" style={{ cursor: 'default' }}>
                              [{idx + 1}]
                            </span>
                            <span>{chunk.source_file}</span>
                            {chunk.section_heading && (
                              <span className="meta-badge">{chunk.section_heading}</span>
                            )}
                          </div>
                          <span className="source-score">
                            Score: {chunk.score.toFixed(4)}
                          </span>
                        </div>
                        <p className="source-snippet">{chunk.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* SIDE SOURCES LIST (Shown only in Single-column layout) */}
        {!compareMode && (
          <div className="pipeline-column">
            <div className="glass-panel">
              <h3 className="panel-title">
                <Database size={18} className="text-purple-400" />
                Retrieved Context Chunks (Ranked)
              </h3>
              
              {!hybridResult && !loading && (
                <div className="text-center text-gray-500 py-8">
                  <Info size={24} className="mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Submit a search query to view source chunks ranked by relevance score.</p>
                </div>
              )}

              {loading && (
                <div className="sources-list">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="source-item">
                      <div className="skeleton skeleton-text" style={{ width: '30%' }} />
                      <div className="skeleton skeleton-text" />
                      <div className="skeleton skeleton-text short" />
                    </div>
                  ))}
                </div>
              )}

              {hybridResult && (
                <div className="sources-list">
                  {hybridResult.retrieved_chunks.map((chunk, idx) => (
                    <div 
                      key={idx} 
                      ref={el => hybridChunkRefs.current[idx + 1] = el}
                      className={`source-item ${highlightedHybridIdx === idx + 1 ? 'highlighted' : ''}`}
                    >
                      <div className="source-item-header">
                        <div className="source-title">
                          <span className="citation-badge" style={{ cursor: 'default' }}>
                            [{idx + 1}]
                          </span>
                          <span>{chunk.source_file}</span>
                          {chunk.section_heading && (
                            <span className="meta-badge">{chunk.section_heading}</span>
                          )}
                        </div>
                        <span className="source-score">
                          Score: {chunk.score.toFixed(4)}
                        </span>
                      </div>
                      <p className="source-snippet">{chunk.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Ingestion & Corpus Explorer Section */}
      <div className="ingest-panel-container">
        {/* Document table list */}
        <div className="glass-panel">
          <h3 className="panel-title">
            <FileText size={18} className="text-purple-400" />
            Indexed Document Workspace
          </h3>
          <table className="document-table">
            <thead>
              <tr>
                <th>Filename</th>
                <th>Format</th>
                <th>Fragments</th>
                <th>File Size</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600 }}>{doc.filename}</td>
                  <td>
                    <span className="meta-badge">{doc.file_type}</span>
                  </td>
                  <td>{doc.fragment_count} sections</td>
                  <td>{(doc.size_bytes / 1024).toFixed(1)} KB</td>
                </tr>
              ))}
              {documents.length === 0 && (
                <tr>
                  <td colSpan={4} className="text-center text-gray-500">
                    No documents indexed. Upload files below to index them.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Upload panel */}
        <div className="glass-panel">
          <h3 className="panel-title">
            <Upload size={18} className="text-teal-400" />
            Ingest New Documentation
          </h3>
          
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: 'none' }}
            accept=".md,.txt,.pdf"
            disabled={uploadLoading}
          />
          
          <div 
            className="file-upload-zone"
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <Upload size={32} className="upload-icon mx-auto" />
            <div className="upload-text">
              {uploadLoading ? 'Ingesting and Indexing...' : 'Click to Upload Document'}
            </div>
            <div className="upload-hint">Supports Markdown (.md), Text (.txt), and PDF (.pdf)</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
