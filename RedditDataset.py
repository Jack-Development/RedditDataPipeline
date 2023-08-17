import os
import glob
from functools import partial
from pathlib import Path
from typing import Tuple, Union

from torchdata.datapipes.iter import FileOpener, IterableWrapper
from torchtext._internal.module_utils import is_module_available
from torchtext.data.datasets_utils import _create_dataset_directory
from torchtext.data.datasets_utils import _wrap_split_argument

import warnings

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data.datapipes.utils', message='Local function is not supported by pickle')

DATASET_NAME = "RedditDataset"
_PATH = ""

def _filter_reddit_data(key, fname):
    # eg. fname = "reddit_data/train/USER_ID/comment_COMMENTID.txt"
    *_, split, _, file_type = Path(fname).parts
    return key == split and (file_type.startswith("comment_") or file_type.startswith("post_"))

def _path_map_fn(t):
    return Path(t[0]).parts[-2], t[1]  # This returns the USER_ID and the content of the file

@_create_dataset_directory(dataset_name=DATASET_NAME)
@_wrap_split_argument(("train", "test"))
def RedditDataset(root: str, split: Union[Tuple[str], str]):
    """RedditDataset

    Args:
        root: Directory where the datasets are saved. Default: os.path.expanduser('~/.torchtext/cache')
        split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)

    :returns: DataPipe that yields tuple of user_id (as a string) and comment/post text
    :rtype: (str, str)
    """
    if not is_module_available("torchdata"):
        raise ModuleNotFoundError(
            "Package `torchdata` not found. Please install following instructions at https://github.com/pytorch/data"
        )

    # Use glob to get a list of all files that match our criteria
    path_pattern = os.path.join(root, _PATH, split, "*", "*.txt")
    all_files = glob.glob(path_pattern)
    
    # Filtering the list based on your criteria
    filtered_files = [f for f in all_files if _filter_reddit_data(split, f)]
    
    # Use IterableWrapper on the filtered list
    files_dp = IterableWrapper(filtered_files)
    
    # Extracting user_id and content
    def extract_content(path):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        user_id = Path(path).parent.name
        return user_id, content

    data_dp = files_dp.map(extract_content)
    
    return data_dp