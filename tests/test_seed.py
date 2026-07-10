from unittest.mock import MagicMock, patch
# pyrefly: ignore [missing-import]
from src.ingestion.seed import main

@patch("src.ingestion.seed.DocumentLoader")
@patch("src.ingestion.seed.ChunkingOrchestrator")
@patch("src.ingestion.seed.IndexManager")
def test_seed_main(mock_index_manager, mock_orchestrator, mock_loader):
    # Setup mocks
    loader_inst = mock_loader.return_value
    orchestrator_inst = mock_orchestrator.return_value
    manager_inst = mock_index_manager.return_value
    
    # Mock loader raw folder path existence check
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    loader_inst.raw_dir = mock_path
    
    # Mock ingest and chunk return values
    loader_inst.ingest_file.return_value = [MagicMock()]
    orchestrator_inst.chunk_documents.return_value = [MagicMock()]
    manager_inst.index_chunks.return_value = (1, 0)
    
    # Run seed main function
    main()
    
    # Verify mock interactions
    manager_inst.clear_indexes.assert_called_once()
    loader_inst.ingest_file.assert_called()
    orchestrator_inst.chunk_documents.assert_called()
    manager_inst.index_chunks.assert_called_once()
