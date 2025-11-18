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
        self.on_upload_callbacks = []  # List of callback functions
        self.metadata = {}
        os.makedirs(self.upload_dir, exist_ok=True)
        
    def add_upload_callback(self, callback):
        """Add a callback function to be called after successful upload"""
        self.on_upload_callbacks.append(callback)
    
    def _trigger_upload_callbacks(self):
        """Trigger all registered upload callbacks"""
        for callback in self.on_upload_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"⚠️ Error in upload callback: {e}")
    
    def upload_file(self, file_path: str, auto_rebuild: bool = True) -> bool:
        """
        Upload a single file to the system.
        
        Args:
            file_path: Path to the file to upload
            auto_rebuild: If True, triggers callbacks after upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return False
            
            # Get filename
            file_id = str(uuid.uuid4()) 
            filename = os.path.basename(file_path)
            destination = os.path.join(self.upload_dir, filename)

            # save metadata
            self.metadata[file_id] = {
                "filename": filename,
                "path": destination
            }
            
            # Copy file to upload directory
            shutil.copy2(file_path, destination)
            print(f"✅ Uploaded: {filename}")
            
            if auto_rebuild and self.on_upload_callbacks:
                self._trigger_upload_callbacks()
                
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
    def upload_multiple_files(self, file_paths: List[str], auto_rebuild: bool = True) -> int:
        """
        Upload multiple files at once.
        
        Args:
            file_paths: List of file paths to upload
            auto_rebuild: If True, triggers callbacks after all files are uploaded
            
        Returns:
            Number of successfully uploaded files
        """
        success_count = 0
        
        print(f"\n📤 Uploading {len(file_paths)} files...")
        
        for file_path in file_paths:
            if self.upload_file(file_path, auto_rebuild=False):  # Don't trigger for each file
                success_count += 1
        
        # Only trigger callbacks once after all files are processed
        if auto_rebuild and success_count > 0 and self.on_upload_callbacks:
            self._trigger_upload_callbacks()
        
        print(f"\n✅ Successfully uploaded {success_count}/{len(file_paths)} files")
        return success_count
    
    def list_uploaded_files(self):
        return [{"id": doc_id, "filename": info["filename"]} for doc_id, info in self.metadata.items()]

    
    def delete_file(self, file_id: str) -> bool:
        if file_id in self.metadata:
            path = self.metadata[file_id]["path"]
            if os.path.exists(path):
                os.remove(path)
            del self.metadata[file_id]
            self.save_metadata()
            print(f"✅ Deleted file with ID: {file_id}")
            return True
        else:
            print(f"❌ File ID not found: {file_id}")
            return False

    
    def clear_all_uploads(self) -> bool:
        """
        Delete all uploaded files using their IDs.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            all_ids = list(self.metadata.keys())
            for file_id in all_ids:
                self.delete_file(file_id)
            print(f"✅ Cleared {len(all_ids)} files")
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