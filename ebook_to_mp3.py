"""Convert Vietnamese ebook text to natural speech MP3 using macOS voices or gTTS."""
from __future__ import annotations

import argparse
import importlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional
import zipfile

gTTS = None  # Lazily loaded to avoid import-time warnings when unused.


class HTMLTextExtractor(HTMLParser):
    """Minimal HTML to plain-text extractor for EPUB XHTML files."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._needs_space = False

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._append("\n")
            self._needs_space = False

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        text = data.strip()
        if not text:
            return
        if self._needs_space:
            self._append(" ")
        self._append(text)
        self._needs_space = True

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"p", "div", "li"}:
            self._append("\n")
            self._needs_space = False

    def _append(self, chunk: str) -> None:
        self._chunks.append(chunk)

    def get_text(self) -> str:
        text = "".join(self._chunks)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


def read_text_from_epub(epub_path: Path) -> str:
    with zipfile.ZipFile(epub_path, "r") as zf:
        html_files = [name for name in zf.namelist() if name.lower().endswith((".xhtml", ".html", ".htm"))]
        if not html_files:
            raise ValueError("Không tìm thấy nội dung XHTML/HTML trong file EPUB.")
        texts: List[str] = []
        for name in sorted(html_files):
            with zf.open(name) as fp:
                raw_bytes = fp.read()
            for encoding in ("utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1"):
                try:
                    data = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                data = raw_bytes.decode("utf-8", errors="ignore")
            extractor = HTMLTextExtractor()
            extractor.feed(data)
            extractor.close()
            page_text = extractor.get_text()
            if page_text:
                texts.append(page_text)
        return "\n\n".join(texts)


def read_text(input_path: Path) -> str:
    if input_path.suffix.lower() == ".epub":
        return read_text_from_epub(input_path)
    encoding_candidates = ["utf-8", "utf-16", "utf-16le", "utf-16be", "latin-1"]
    raw_bytes = input_path.read_bytes()
    for encoding in encoding_candidates:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def sanitize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class ConversionOptions:
    engine: str
    voice: Optional[str]
    rate: Optional[int]
    prefer: List[str]


def ensure_command_available(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f"Không tìm thấy lệnh '{command}'.")


def run(command: List[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Lệnh {' '.join(command)} thất bại với mã {exc.returncode}.") from exc


def call_say(text_file: Path, aiff_file: Path, options: ConversionOptions) -> None:
    ensure_command_available("say")
    cmd: List[str] = ["say", "-f", str(text_file), "-o", str(aiff_file)]
    if options.voice:
        cmd.extend(["-v", options.voice])
    if options.rate:
        cmd.extend(["-r", str(options.rate)])
    run(cmd)


def convert_audio(aiff_file: Path, output_file: Path, options: ConversionOptions) -> None:
    converters = options.prefer or ["ffmpeg", "afconvert"]
    converters = [c.lower() for c in converters]
    attempted: List[str] = []
    for converter in converters:
        if converter == "ffmpeg" and shutil.which("ffmpeg"):
            command = [
                "ffmpeg",
                "-y",
                "-i",
                str(aiff_file),
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "2",
                str(output_file),
            ]
            try:
                run(command)
                return
            except RuntimeError as exc:
                attempted.append(f"ffmpeg (lỗi: {exc})")
        elif converter == "afconvert" and shutil.which("afconvert"):
            command = [
                "afconvert",
                str(aiff_file),
                str(output_file),
                "-f",
                "MPG3",
                "-d",
                ".mp3",
                "-b",
                "192000",
            ]
            try:
                run(command)
                return
            except RuntimeError as exc:
                attempted.append(f"afconvert (lỗi: {exc})")
        else:
            attempted.append(f"{converter} (không khả dụng)")
    attempted_msg = "; ".join(attempted)
    raise RuntimeError(
        "Không thể tạo file MP3. "
        "Vui lòng cài đặt 'ffmpeg' (Homebrew: brew install ffmpeg) hoặc kiểm tra afconvert hỗ trợ mã hóa MP3. "
        f"Đã thử: {attempted_msg}"
    )


def list_voices() -> None:
    ensure_command_available("say")
    subprocess.run(["say", "-v", "?"], check=True)


def split_text_for_gtts(text: str, max_chars: int = 4500) -> List[str]:
    """Split long input into chunks within Google TTS length limits."""

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0

    def flush() -> None:
        if buffer:
            chunks.append(" ".join(buffer).strip())

    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            flush()
            buffer.clear()
            current_len = 0
            sentences = textwrap.wrap(paragraph, max_chars, break_long_words=False, break_on_hyphens=False)
            for sentence in sentences:
                if len(sentence) > max_chars:
                    # Fallback to hard split if still too long.
                    pieces = [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
                else:
                    pieces = [sentence]
                for piece in pieces:
                    chunks.append(piece.strip())
            continue

        candidate_len = current_len + len(paragraph) + (1 if buffer else 0)
        if candidate_len > max_chars:
            flush()
            buffer = [paragraph]
            current_len = len(paragraph)
        else:
            buffer.append(paragraph)
            current_len = candidate_len

    flush()
    return [chunk for chunk in chunks if chunk]


def synthesize_with_gtts(text: str, output_file: Path) -> None:
    global gTTS
    if gTTS is None:
        try:
            gtts_module = importlib.import_module("gtts")
        except ImportError as exc:
            raise RuntimeError("Chưa cài đặt gTTS. Vui lòng chạy 'pip install gTTS'.") from exc
        gTTS = getattr(gtts_module, "gTTS")

    chunks = split_text_for_gtts(text)
    if not chunks:
        raise RuntimeError("Không có nội dung để tổng hợp bằng gTTS.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        for index, chunk in enumerate(chunks, start=1):
            temp_mp3 = tmp_dir / f"chunk_{index:04d}.mp3"
            tts = gTTS(text=chunk, lang="vi")
            tts.save(str(temp_mp3))
            with temp_mp3.open("rb") as src, output_file.open("ab") as dst:
                shutil.copyfileobj(src, dst)


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chuyển sách điện tử tiếng Việt thành file MP3 với giọng đọc tự nhiên (macOS 'say' hoặc gTTS).")
    parser.add_argument("input", type=Path, nargs="?", help="Đường dẫn đến ebook (.txt, .epub)")
    parser.add_argument("output", type=Path, nargs="?", help="Đường dẫn file .mp3 đầu ra")
    parser.add_argument("--engine", choices=["say", "gtts"], default="say", help="Chọn bộ máy tổng hợp giọng (mặc định: say).")
    parser.add_argument("--voice", help="Tên giọng đọc (chỉ áp dụng với engine 'say').")
    parser.add_argument("--rate", type=int, help="Tốc độ đọc (từ/phút, chỉ áp dụng với 'say').")
    parser.add_argument(
        "--prefer",
        nargs="*",
        choices=["ffmpeg", "afconvert"],
        help="Ưu tiên bộ chuyển đổi âm thanh (mặc định: ffmpeg rồi afconvert).",
    )
    parser.add_argument("--list-voices", action="store_true", help="Hiển thị danh sách giọng đọc và thoát")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    if args.list_voices:
        if args.engine != "say":
            print("--list-voices chỉ hỗ trợ với engine 'say'.", file=sys.stderr)
            return 2
        list_voices()
        return 0
    if args.input is None or args.output is None:
        print("Yêu cầu chỉ định đường dẫn input và output, trừ khi dùng --list-voices.", file=sys.stderr)
        return 2

    input_path: Path = args.input
    output_path: Path = args.output
    engine = args.engine
    if engine == "gtts" and (args.voice or args.rate):
        print("Cảnh báo: --voice và --rate chỉ áp dụng với engine 'say'.", file=sys.stderr)
    if not output_path.parent.exists():
        print(f"Thư mục đầu ra không tồn tại: {output_path.parent}", file=sys.stderr)
        return 1
    if not input_path.exists():
        print(f"Không tìm thấy file đầu vào: {input_path}", file=sys.stderr)
        return 1
    try:
        raw_text = read_text(input_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Lỗi đọc file: {exc}", file=sys.stderr)
        return 1
    text = sanitize_text(raw_text)
    if not text:
        print("File đầu vào không có nội dung hợp lệ.", file=sys.stderr)
        return 1
    options = ConversionOptions(engine=engine, voice=args.voice, rate=args.rate, prefer=args.prefer or [])

    if engine == "say":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            tmp_text = tmp_dir / "input.txt"
            tmp_aiff = tmp_dir / "voice.aiff"
            tmp_text.write_text(text, encoding="utf-8")
            try:
                call_say(tmp_text, tmp_aiff, options)
                convert_audio(tmp_aiff, output_path, options)
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 1
    elif engine == "gtts":
        try:
            synthesize_with_gtts(text, output_path)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    else:  # pragma: no cover - guarded by argparse choices
        print(f"Engine không được hỗ trợ: {engine}", file=sys.stderr)
        return 2
    print(f"Đã tạo file MP3: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
