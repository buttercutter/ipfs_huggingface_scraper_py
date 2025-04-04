"""Test the CLI component."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import argparse
from ipfs_huggingface_scraper_py.cli import parse_args, setup_logging, init_config, show_status, main

@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    # Create a temporary file
    fd, config_path = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    
    with open(config_path, 'w') as f:
        f.write("""
[scraper]
output_dir = "test_output"
max_models = 10
        """)
    
    yield config_path
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)

def test_parse_args():
    """Test argument parsing."""
    # Test scrape command
    with patch('sys.argv', ['hf-scraper', 'scrape', '--config', 'config.toml', '--max-models', '10']):
        args = parse_args()
        assert args.command == 'scrape'
        assert args.config == 'config.toml'
        assert args.max_models == 10
    
    # Test resume command
    with patch('sys.argv', ['hf-scraper', 'resume', '--config', 'config.toml']):
        args = parse_args()
        assert args.command == 'resume'
        assert args.config == 'config.toml'
    
    # Test init command
    with patch('sys.argv', ['hf-scraper', 'init', '--output', 'my_config.toml']):
        args = parse_args()
        assert args.command == 'init'
        assert args.output == 'my_config.toml'
    
    # Test status command
    with patch('sys.argv', ['hf-scraper', 'status', '--config', 'config.toml']):
        args = parse_args()
        assert args.command == 'status'
        assert args.config == 'config.toml'
    
    # Test log level
    with patch('sys.argv', ['hf-scraper', 'scrape', '--log-level', 'DEBUG']):
        args = parse_args()
        assert args.log_level == 'DEBUG'

def test_setup_logging():
    """Test logging setup."""
    with patch('logging.basicConfig') as mock_basic_config:
        # Test default level
        setup_logging()
        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == logging.INFO
        
        # Test custom level
        mock_basic_config.reset_mock()
        setup_logging("DEBUG")
        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == logging.DEBUG

def test_init_config():
    """Test config initialization."""
    # Create a temporary output file
    fd, output_path = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    os.remove(output_path)  # Remove so init_config can create it
    
    try:
        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            # Initialize config
            init_config(output_path)
            
            # Check that file was created
            assert os.path.exists(output_path)
            
            # Check that success message was printed
            mock_print.assert_any_call(f"Configuration template created at: {output_path}")
    finally:
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)
    
    # Test with error
    with patch('ipfs_huggingface_scraper_py.config.Config.export_config_template', 
              side_effect=Exception("Config error")):
        with patch('builtins.print') as mock_print:
            with patch('sys.exit') as mock_exit:
                # Initialize config
                init_config("nonexistent/path.toml")
                
                # Check that error message was printed
                mock_print.assert_any_call("Error creating configuration template: Config error")
                
                # Check that exit was called
                mock_exit.assert_called_once_with(1)

def test_show_status(temp_config_file):
    """Test status display."""
    # Mock state manager and progress
    mock_state_manager = MagicMock()
    mock_state_manager.get_progress.return_value = {
        "is_completed": True,
        "is_paused": False,
        "total_models_discovered": 100,
        "models_processed": 75,
        "models_completed": 70,
        "models_errored": 5,
        "completion_percentage": 75.0,
        "elapsed_time": 3665,  # 1h 1m 5s
        "current_position": 75
    }
    
    # Patch state manager class
    with patch('ipfs_huggingface_scraper_py.state_manager.StateManager', return_value=mock_state_manager):
        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            # Show status
            show_status(temp_config_file)
            
            # Check output
            mock_print.assert_any_call("=== Scraper Status ===")
            mock_print.assert_any_call("Status: Completed")
            mock_print.assert_any_call("Total models discovered: 100")
            mock_print.assert_any_call("Models processed: 75")
            mock_print.assert_any_call("Models completed: 70")
            mock_print.assert_any_call("Models with errors: 5")
            mock_print.assert_any_call("Completion: 75.00%")
            mock_print.assert_any_call("Elapsed time: 1h 1m 5s")
            mock_print.assert_any_call("Current position: 75")
    
    # Test with error
    with patch('ipfs_huggingface_scraper_py.state_manager.StateManager', 
              side_effect=Exception("State error")):
        with patch('builtins.print') as mock_print:
            with patch('sys.exit') as mock_exit:
                # Show status
                show_status(temp_config_file)
                
                # Check that error message was printed
                mock_print.assert_any_call("Error showing status: State error")
                
                # Check that exit was called
                mock_exit.assert_called_once_with(1)

def test_main():
    """Test main function."""
    # Test scrape command
    with patch('sys.argv', ['hf-scraper', 'scrape']):
        with patch('ipfs_huggingface_scraper_py.cli.EnhancedScraper') as mock_scraper_class:
            mock_scraper = MagicMock()
            mock_scraper_class.return_value = mock_scraper
            
            # Run main
            main()
            
            # Check that scraper was created and used
            mock_scraper_class.assert_called_once()
            mock_scraper.scrape_models.assert_called_once_with(None)
    
    # Test scrape command with max_models
    with patch('sys.argv', ['hf-scraper', 'scrape', '--max-models', '10']):
        with patch('ipfs_huggingface_scraper_py.cli.EnhancedScraper') as mock_scraper_class:
            mock_scraper = MagicMock()
            mock_scraper_class.return_value = mock_scraper
            
            # Run main
            main()
            
            # Check that scraper was created and used
            mock_scraper_class.assert_called_once()
            mock_scraper.scrape_models.assert_called_once_with(10)
    
    # Test resume command
    with patch('sys.argv', ['hf-scraper', 'resume']):
        with patch('ipfs_huggingface_scraper_py.cli.EnhancedScraper') as mock_scraper_class:
            mock_scraper = MagicMock()
            mock_scraper_class.return_value = mock_scraper
            
            # Run main
            main()
            
            # Check that scraper was created and used
            mock_scraper_class.assert_called_once()
            mock_scraper.resume.assert_called_once()
    
    # Test init command
    with patch('sys.argv', ['hf-scraper', 'init']):
        with patch('ipfs_huggingface_scraper_py.cli.init_config') as mock_init_config:
            # Run main
            main()
            
            # Check that init_config was called
            mock_init_config.assert_called_once_with('config.toml')
    
    # Test status command
    with patch('sys.argv', ['hf-scraper', 'status']):
        with patch('ipfs_huggingface_scraper_py.cli.show_status') as mock_show_status:
            # Run main
            main()
            
            # Check that show_status was called
            mock_show_status.assert_called_once_with(None)
    
    # Test no command
    with patch('sys.argv', ['hf-scraper']):
        with patch('builtins.print') as mock_print:
            with patch('sys.exit') as mock_exit:
                # Run main
                main()
                
                # Check that help message was printed
                mock_print.assert_any_call("No command specified. Use one of: scrape, resume, init, status")
                
                # Check that exit was called
                mock_exit.assert_called_once_with(1)

# Import logging here instead of at the top to avoid interfering with the tests
import logging