# Ebook to MP3

Công cụ dòng lệnh chuyển ebook tiếng Việt (.txt hoặc .epub) thành file MP3 bằng giọng đọc tự nhiên. Hỗ trợ:

- `say` (giọng hệ thống macOS, gồm Siri Voice, Linh, v.v.).
- `gTTS` (Google Text-to-Speech, cần kết nối Internet).
- Tách ebook thành nhiều file theo chương hoặc theo thời lượng mong muốn.

## Yêu cầu

- Python 3.9+.
- macOS để dùng engine `say`.
- Internet và thư viện `gTTS` khi dùng engine `gtts`:
  ```bash
  pip install gTTS
  ```
- Nếu xuất MP3 bằng `say`, nên có `ffmpeg` hoặc đảm bảo `afconvert` hỗ trợ MP3.

## Cách sử dụng cơ bản

```bash
python3 ebook_to_mp3.py <input> <output.mp3> [tùy chọn]
```

Ví dụ nhanh với gTTS:

```bash
python3 ebook_to_mp3.py text.txt text.mp3 --engine gtts
```

Ví dụ dùng giọng Siri trên macOS:

```bash
python3 ebook_to_mp3.py ebook.txt ebook.mp3 --engine say --voice "Siri (Vietnamese (Voice 1))" --rate 170
```

Liệt kê các giọng `say` có sẵn:

```bash
python3 ebook_to_mp3.py --engine say --list-voices
```

## Chia nhỏ ebook

### Theo chương

- EPUB: chương được lấy tự động từ từng file `.xhtml`/`.html`.
- TXT: cung cấp regex nhận diện tiêu đề chương, ví dụ nếu mỗi chương bắt đầu bằng "Chương 1", "Chương 2"…

```bash
python3 ebook_to_mp3.py novel.txt output.mp3 \
  --engine gtts \
  --split-mode sections \
  --split-marker "^Chương\\s+\\d+"
```

Các file sẽ được tạo thành `output_001.mp3`, `output_002.mp3`, …

### Theo thời lượng ước lượng

Script ước lượng số từ dựa trên tốc độ đọc (`--words-per-minute`, mặc định 160 wpm) và tách thành các đoạn gần với thời lượng yêu cầu.

```bash
python3 ebook_to_mp3.py book.epub book.mp3 \
  --split-mode duration \
  --target-seconds 900 \
  --words-per-minute 170
```

## Tùy chọn quan trọng

| Tham số                                 | Mô tả                                                             |
| --------------------------------------- | ----------------------------------------------------------------- |
| `--engine {say,gtts}`                   | Chọn bộ tổng hợp giọng (mặc định `say`).                          |
| `--voice`                               | Tên giọng dùng cho `say` (ví dụ `"Siri (Vietnamese (Voice 1))"`). |
| `--rate`                                | Tốc độ đọc (từ/phút) với engine `say`.                            |
| `--prefer`                              | Thứ tự ưu tiên khi chuyển AIFF sang MP3 (`ffmpeg`, `afconvert`).  |
| `--split-mode {none,sections,duration}` | Cơ chế tách thành nhiều file.                                     |
| `--split-marker`                        | Regex đánh dấu chương cho file TXT khi dùng `sections`.           |
| `--target-seconds`                      | Thời lượng mục tiêu mỗi file khi dùng `duration`.                 |
| `--words-per-minute`                    | Tốc độ đọc ước lượng để tính thời lượng (mặc định 160).           |

## Ghi chú

- Engine `gtts` cần Internet; nếu mất kết nối, script hủy file MP3 dở dang và báo lỗi.
- Các file MP3 đầu ra sẽ được đặt trong cùng thư mục với `output` đã chỉ định.
- Khi tách nhiều phần, `output` được hiểu là tiền tố; ví dụ `audiobook.mp3` -> `audiobook_001.mp3`, `audiobook_002.mp3`...

python3 ebook_to_mp3.py book.epub book.mp3 \
 --engine gtts \
 --split-mode duration \
 --target-seconds 900 \
 --words-per-minute 238
