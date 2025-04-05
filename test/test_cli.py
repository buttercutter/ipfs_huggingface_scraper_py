"""Test the CLI component."""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import argparse
import sys
from ipfs_huggingface_scraper_py.cli import main

def test_cli_module_structure():
    """Test that the CLI module has the necessary components."""
    import ipfs_huggingface_scraper_py.cli as cli
    
    # Check that the main function exists
    assert hasattr(cli, "main")
    assert callable(cli.main)

@patch('sys.argv', ['hf-scraper'])
@patch('sys.exit')
def test_main_no_command(mock_exit):
    """Test main function with no command."""
    # When no command is provided, it should show help and exit
    with patch('sys.argv', ['hf-scraper']):
        with patch('argparse.ArgumentParser.print_help'):
            main()
            mock_exit.assert_called_once_with(1)

@patch('ipfs_huggingface_scraper_py.cli.EnhancedScraper')
def test_main_models_command(mock_scraper_class):
    """Test models command."""
    mock_scraper = MagicMock()
    mock_scraper_class.return_value = mock_scraper
    
    with patch('sys.argv', ['hf-scraper', 'models', '--max', '10']):
        with patch('sys.exit'):
            main()
            mock_scraper_class.assert_called_once()
            mock_scraper.scrape_models.assert_called_once_with(10)

@patch('ipfs_huggingface_scraper_py.cli.DatasetsScraper')
def test_main_datasets_command(mock_scraper_class):
    """Test datasets command."""
    mock_scraper = MagicMock()
    mock_scraper_class.return_value = mock_scraper
    
    with patch('sys.argv', ['hf-scraper', 'datasets', '--max', '10']):
        with patch('sys.exit'):
            main()
            mock_scraper_class.assert_called_once()
            mock_scraper.scrape_datasets.assert_called_once_with(10)

@patch('ipfs_huggingface_scraper_py.cli.SpacesScraper')
def test_main_spaces_command(mock_scraper_class):
    """Test spaces command."""
    mock_scraper = MagicMock()
    mock_scraper_class.return_value = mock_scraper
    
    with patch('sys.argv', ['hf-scraper', 'spaces', '--max', '10']):
        with patch('sys.exit'):
            main()
            mock_scraper_class.assert_called_once()
            mock_scraper.scrape_spaces.assert_called_once_with(10)

@patch('ipfs_huggingface_scraper_py.cli.EnhancedScraper')
@patch('ipfs_huggingface_scraper_py.cli.DatasetsScraper')
@patch('ipfs_huggingface_scraper_py.cli.SpacesScraper')
def test_main_all_command(mock_spaces_class, mock_datasets_class, mock_models_class):
    """Test all command."""
    mock_models = MagicMock()
    mock_datasets = MagicMock()
    mock_spaces = MagicMock()
    
    mock_models_class.return_value = mock_models
    mock_datasets_class.return_value = mock_datasets
    mock_spaces_class.return_value = mock_spaces
    
    with patch('sys.argv', ['hf-scraper', 'all', '--max-models', '10', '--max-datasets', '5', '--max-spaces', '3']):
        with patch('sys.exit'):
            main()
            mock_models_class.assert_called_once()
            mock_datasets_class.assert_called_once()
            mock_spaces_class.assert_called_once()
            
            mock_models.scrape_models.assert_called_once()
            mock_datasets.scrape_datasets.assert_called_once()
            mock_spaces.scrape_spaces.assert_called_once()

@patch('ipfs_huggingface_scraper_py.cli.Config')
def test_export_config_command(mock_config_class):
    """Test export-config command."""
    mock_config = MagicMock()
    mock_config_class.return_value = mock_config
    
    with tempfile.NamedTemporaryFile(suffix='.toml') as temp_file:
        with patch('sys.argv', ['hf-scraper', 'export-config', '--output', temp_file.name]):
            with patch('sys.exit'):
                main()
                mock_config.export_config_template.assert_called_once_with(temp_file.name)

def test_resume_commands():
    """Test resume commands for different entity types."""
    # Models resume
    with patch('ipfs_huggingface_scraper_py.cli.EnhancedScraper') as mock_scraper_class:
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper
        
        with patch('sys.argv', ['hf-scraper', 'models', '--resume']):
            with patch('sys.exit'):
                main()
                mock_scraper_class.assert_called_once()
                mock_scraper.resume.assert_called_once()
    
    # Datasets resume
    with patch('ipfs_huggingface_scraper_py.cli.DatasetsScraper') as mock_scraper_class:
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper
        
        with patch('sys.argv', ['hf-scraper', 'datasets', '--resume']):
            with patch('sys.exit'):
                main()
                mock_scraper_class.assert_called_once()
                mock_scraper.resume.assert_called_once()
    
    # Spaces resume
    with patch('ipfs_huggingface_scraper_py.cli.SpacesScraper') as mock_scraper_class:
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper
        
        with patch('sys.argv', ['hf-scraper', 'spaces', '--resume']):
            with patch('sys.exit'):
                main()
                mock_scraper_class.assert_called_once()
                mock_scraper.resume.assert_called_once()

def test_token_handling():
    """Test API token handling."""
    with patch('ipfs_huggingface_scraper_py.cli.EnhancedScraper') as mock_scraper_class:
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper
        
        with patch('sys.argv', ['hf-scraper', 'models', '--token', 'test_token']):
            with patch('os.environ') as mock_environ:
                with patch('sys.exit'):
                    main()
                    mock_environ.__setitem__.assert_called_with('HF_API_TOKEN', 'test_token')