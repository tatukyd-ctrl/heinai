# backend/dropbox_utils.py
import os
import dropbox
from dropbox.exceptions import ApiError
from backend.config import DROPBOX_ACCESS_TOKEN, DROPBOX_BASE_FOLDER

def _normalize_dropbox_path(folder: str, filename: str) -> str:
    # ensure leading slash, no double slashes
    if not folder:
        folder = "/"
    if not folder.startswith("/"):
        folder = "/" + folder
    if folder.endswith("/"):
        folder = folder[:-1]
    return f"{folder}/{filename}"

def upload_to_dropbox(file_path: str, folder: str = None) -> str:
    """
    Upload local file to Dropbox. Returns a shared link (https).
    """
    if not DROPBOX_ACCESS_TOKEN:
        raise RuntimeError("DROPBOX_ACCESS_TOKEN not set in env")
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

    target_folder = folder if folder is not None else DROPBOX_BASE_FOLDER or "/"
    target_path = _normalize_dropbox_path(target_folder, os.path.basename(file_path))

    with open(file_path, "rb") as f:
        data = f.read()
    # upload (overwrite)
    try:
        dbx.files_upload(data, target_path, mode=dropbox.files.WriteMode("overwrite"))
    except ApiError as e:
        # propagate meaningful error
        raise RuntimeError(f"Dropbox upload failed: {e}")

    # get or create shared link
    try:
        res = dbx.sharing_create_shared_link_with_settings(target_path)
        return res.url
    except ApiError:
        # maybe link already exists — retrieve it
        try:
            links = dbx.sharing_list_shared_links(path=target_path, direct_only=True).links
            if links:
                return links[0].url
        except ApiError:
            pass
        raise RuntimeError("Failed to create or fetch Dropbox shared link")
