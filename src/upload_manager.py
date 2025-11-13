"""
Module for handling document uploads and management.
"""

import os
import shutil
from pathlib import Path
from typing import List
from src.config import Config

class UploadManager:
    """
    Manages document uploads and file operations.
    """
    
    def __init__(self):
        self.upload_dir = Config.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def upload_file(self, file_path: str) -> bool:
        """
        Upload a single file to the system.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return False
            
            # Get filename
            filename = os.path.basename(file_path)
            destination = os.path.join(self.upload_dir, filename)
            
            # Copy file to upload directory
            shutil.copy2(file_path, destination)
            print(f"✅ Uploaded: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def upload_multiple_files(self, file_paths: List[str]) -> int:
        """
        Upload multiple files at once.
        
        Args:
            file_paths: List of file paths to upload
            
        Returns:
            Number of successfully uploaded files
        """
        success_count = 0
        
        print(f"\n📤 Uploading {len(file_paths)} files...")
        
        for file_path in file_paths:
            if self.upload_file(file_path):
                success_count += 1
        
        print(f"\n✅ Successfully uploaded {success_count}/{len(file_paths)} files")
        return success_count
    
    def list_uploaded_files(self) -> List[str]:
        """
        List all uploaded files.
        
        Returns:
            List of uploaded filenames
        """
        if not os.path.exists(self.upload_dir):
            return []
        
        # Get all files (excluding directories and hidden files)
        files = [f for f in os.listdir(self.upload_dir) 
                if os.path.isfile(os.path.join(self.upload_dir, f)) and not f.startswith('.')]
        return files
    
    def delete_file(self, filename: str) -> bool:
        """
        Delete an uploaded file.
        
        Args:
            filename: Name of file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Deleted: {filename}")
                return True
            else:
                print(f"❌ File not found: {filename}")
                return False
        except Exception as e:
            print(f"❌ Delete failed: {e}")
            return False
    
    def clear_all_uploads(self) -> bool:
        """
        Delete all uploaded files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            files = self.list_uploaded_files()
            for filename in files:
                self.delete_file(filename)
            print(f"✅ Cleared {len(files)} files")
            return True
        except Exception as e:
            print(f"❌ Clear failed: {e}")
            return False
    
    def get_upload_dir(self) -> str:
        """
        Get the upload directory path.
        
        Returns:
            Path to upload directory
        """
        return self.upload_dir