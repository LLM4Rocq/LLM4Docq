from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, Iterator, List, Tuple, Optional


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int
    dev: int
    ino: int


def iter_v_files(root: str) -> Iterator[FileInfo]:
    """
    Fast directory walk using os.scandir (typically faster than os.walk).
    Yields FileInfo for regular files.
    """
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir():
                            stack.append(entry.path)
                        elif entry.is_file():
                            st = entry.stat()
                            if entry.name == 'source.v':
                                yield FileInfo(entry.path, st.st_size, st.st_dev, st.st_ino)
                    except (OSError, PermissionError):
                        # Skip entries we can't stat/read
                        continue
        except (OSError, PermissionError):
            continue


def hash_file_full(path: str, algo: str = "blake2b", chunk_size: int = 1024 * 1024) -> str:
    """
    Full-file hash (streaming). blake2b is fast and in hashlib.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_file_quick(
    path: str,
    size: int,
    algo: str = "blake2b",
    sample_bytes: int = 64 * 1024,
) -> str:
    """
    Quick fingerprint: hash first sample_bytes and last sample_bytes (if file is big enough),
    plus the size (size is included indirectly in grouping but adding it doesn't hurt).
    """
    h = hashlib.new(algo, digest_size=16) if algo == "blake2b" else hashlib.new(algo)
    h.update(size.to_bytes(8, "little", signed=False))

    with open(path, "rb") as f:
        head = f.read(sample_bytes)
        h.update(head)

        if size > sample_bytes:
            # Read tail
            tail_size = min(sample_bytes, size)
            f.seek(-tail_size, os.SEEK_END)
            tail = f.read(tail_size)
            h.update(tail)

    return h.hexdigest()


def files_equal(a: str, b: str, chunk_size: int = 1024 * 1024) -> bool:
    """
    Byte-by-byte chunk compare. Guaranteed correct.
    """
    if a == b:
        return True
    try:
        sa = os.stat(a)
        sb = os.stat(b)
        if sa.st_size != sb.st_size:
            return False
    except OSError:
        return False

    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ba = fa.read(chunk_size)
            bb = fb.read(chunk_size)
            if ba != bb:
                return False
            if not ba:
                return True


def find_source_duplicates(
    root: str,
    workers: int = 8,
    algo: str = "blake2b",
    sample_bytes: int = 64 * 1024,
) -> List[List[str]]:
    """
    Returns a list of groups, each group is a list of file paths that are identical.
    """
    # 1) Collect + group by size
    by_size: Dict[int, List[FileInfo]] = defaultdict(list)
    all_files: List[FileInfo] = []

    for fi in iter_v_files(root):
        by_size[fi.size].append(fi)
        all_files.append(fi)
    
    if not all_files:
        return []
    
    size_candidates = {sz: lst for sz, lst in by_size.items() if len(lst) > 1}

    if not size_candidates:
        return [[fi.path] for fi in all_files]

    # 2) Quick fingerprint within same-size groups
    by_quick: Dict[Tuple[int, str], List[FileInfo]] = defaultdict(list)

    def _quick(fi: FileInfo) -> Tuple[FileInfo, Optional[str]]:
        try:
            return fi, hash_file_quick(fi.path, fi.size, algo=algo, sample_bytes=sample_bytes)
        except OSError:
            return fi, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_quick, fi)
                   for files in size_candidates.values()
                   for fi in files]

        for fut in as_completed(futures):
            fi, qh = fut.result()
            if qh is not None:
                by_quick[(fi.size, qh)].append(fi)

    quick_candidates = {k: v for k, v in by_quick.items() if len(v) > 1}

    if not quick_candidates:
        return [[fi.path] for fi in all_files]

    # 3) Full hash
    by_full: Dict[str, List[FileInfo]] = defaultdict(list)

    def _full(fi: FileInfo) -> Tuple[FileInfo, Optional[str]]:
        try:
            return fi, hash_file_full(fi.path, algo=algo)
        except OSError:
            return fi, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_full, fi)
                   for files in quick_candidates.values()
                   for fi in files]

        for fut in as_completed(futures):
            fi, fh = fut.result()
            if fh is not None:
                by_full[fh].append(fi)

    # 4) Verify equality
    duplicates: List[List[str]] = []
    duplicate_paths: Set[str] = set()

    for group in by_full.values():
        if len(group) < 2:
            continue

        ref = group[0].path
        confirmed = [ref]

        for fi in group[1:]:
            if files_equal(ref, fi.path):
                confirmed.append(fi.path)

        if len(confirmed) > 1:
            duplicates.append(confirmed)
            duplicate_paths.update(confirmed)

    # 5) Add singletons
    for fi in all_files:
        if fi.path not in duplicate_paths:
            duplicates.append([fi.path])

    return duplicates