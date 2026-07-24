"""
Shared fixtures for the smoke tests.

Fixtures are generated on the fly rather than committed as binaries: a
repo destined for an air-gapped review is easier to trust when it has no
opaque media files in it, and generating them means the test asserts on
text we chose, so "did OCR actually read this?" has a real answer instead
of a vague non-empty check.

Audio and video need macOS tooling (`say`, `afconvert`) or ffmpeg; those
fixtures skip cleanly when the tooling is absent rather than failing.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))

# Phrases planted in each fixture so a test can prove the pipeline read the
# real content rather than returning something plausible-looking.
IMAGE_PHRASES = ["DRY DOCK 12", "NNS-2026-0847"]
AUDIO_PHRASES = ["operations briefing", "welding"]
SPOKEN_TEXT = (
    "Good morning. This is the daily operations briefing for dry dock "
    "twelve. All welding work is scheduled to complete by Friday."
)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def image_file(fixture_dir: Path) -> Path:
    """A synthetic scanned safety notice, including a key/value block."""
    PIL = pytest.importorskip("PIL", reason="pillow is needed to build the image fixture")
    from PIL import Image, ImageDraw, ImageFont

    path = fixture_dir / "safety_notice.png"
    image = Image.new("RGB", (1000, 420), "white")
    draw = ImageDraw.Draw(image)
    try:
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
        body = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 30)
    except OSError:  # No Arial (most Linux CI images) — default font still OCRs.
        bold = body = ImageFont.load_default()

    draw.text((40, 40), "SAFETY NOTICE - DRY DOCK 12", font=bold, fill="black")
    draw.text((40, 120), "Hard hats and steel-toe boots required.", font=body, fill="black")
    draw.text((40, 170), "Report incidents to Safety Officer ext. 4471.", font=body, fill="black")
    draw.text((40, 220), "Permit Number: NNS-2026-0847", font=body, fill="black")
    draw.text((40, 270), "Inspection Date: 2026-07-15", font=body, fill="black")
    image.save(path)
    return path


@pytest.fixture(scope="session")
def textless_photo(fixture_dir: Path) -> Path:
    """
    A picture with no text anywhere in it.

    This is the regression fixture for a real failure: OCR on a photograph
    returns nothing, so routing images through a document-extraction
    service made "what is this image about" fail on any photo that happened
    not to contain writing.
    """
    pytest.importorskip("PIL", reason="pillow is needed to build the photo fixture")
    from PIL import Image, ImageDraw

    path = fixture_dir / "tower_photo.jpg"
    image = Image.new("RGB", (700, 900), "#87CEEB")
    draw = ImageDraw.Draw(image)
    for y in range(900):
        draw.line(
            [(0, y), (700, y)],
            fill=(135 + int(y * 0.06), 206 - int(y * 0.08), 235 - int(y * 0.05)),
        )
    draw.polygon([(350, 80), (300, 400), (400, 400)], fill="#E8532B")
    draw.polygon([(300, 400), (240, 760), (460, 760), (400, 400)], fill="#E8532B")
    draw.rectangle([(0, 760), (700, 900)], fill="#4a4a4a")
    image.save(path, "JPEG")
    return path


# PII planted in a fixture so a test can prove redaction fired end-to-end
# through a real extractor (vision / BDA), not just on a .txt. The name and
# email are the reliable signals; the shipyard line must survive.
PII_NAME = "Maria Gonzalez"
PII_EMAIL = "maria.gonzalez@example.com"
PII_SURVIVES = "Dry Dock 12"


@pytest.fixture(scope="session")
def pii_image_file(fixture_dir: Path) -> Path:
    """A synthetic interview note containing a name, email, and SSN."""
    pytest.importorskip("PIL", reason="pillow is needed to build the PII image fixture")
    from PIL import Image, ImageDraw, ImageFont

    path = fixture_dir / "interview_note.png"
    image = Image.new("RGB", (1000, 360), "white")
    draw = ImageDraw.Draw(image)
    try:
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
        body = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    except OSError:
        bold = body = ImageFont.load_default()

    draw.text((40, 30), "INTERVIEW NOTE", font=bold, fill="black")
    draw.text((40, 100), f"Interviewee: {PII_NAME}", font=body, fill="black")
    draw.text((40, 150), f"Email: {PII_EMAIL}", font=body, fill="black")
    draw.text((40, 200), "SSN: 457-55-5462", font=body, fill="black")
    draw.text((40, 250), f"Re: welding schedule for {PII_SURVIVES}.", font=body, fill="black")
    image.save(path)
    return path


@pytest.fixture(scope="session")
def pii_pdf_file(fixture_dir: Path, pii_image_file: Path) -> Path:
    """The same PII note as a PDF, to exercise the BDA document branch."""
    pytest.importorskip("PIL", reason="pillow is needed to build the PII PDF fixture")
    from PIL import Image

    path = fixture_dir / "interview_note.pdf"
    Image.open(pii_image_file).convert("RGB").save(path, "PDF")
    return path


@pytest.fixture(scope="session")
def audio_file(fixture_dir: Path) -> Path:
    """Real synthesized speech, so the transcript assertion means something."""
    if not (shutil.which("say") and shutil.which("afconvert")):
        pytest.skip("`say`/`afconvert` (macOS) needed to build the audio fixture")

    aiff, wav = fixture_dir / "speech.aiff", fixture_dir / "speech.wav"
    subprocess.run(["say", "-o", str(aiff), SPOKEN_TEXT], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
        check=True,
    )
    aiff.unlink()
    return wav


@pytest.fixture(scope="session")
def pdf_file(fixture_dir: Path, image_file: Path) -> Path:
    """The same notice as a PDF, to cover the document branch."""
    PIL = pytest.importorskip("PIL", reason="pillow is needed to build the PDF fixture")
    from PIL import Image

    path = fixture_dir / "safety_notice.pdf"
    Image.open(image_file).convert("RGB").save(path, "PDF")
    return path


def _ffmpeg() -> str | None:
    """System ffmpeg, else the binary imageio-ffmpeg bundles."""
    if system := shutil.which("ffmpeg"):
        return system
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:  # No prebuilt binary for this platform.
        return None


@pytest.fixture(scope="session")
def video_file(fixture_dir: Path, image_file: Path) -> Path:
    """A few seconds of the notice on screen — needs ffmpeg."""
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg needed to build the video fixture (pip install imageio-ffmpeg)")

    path = fixture_dir / "safety_notice.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loop", "1", "-i", str(image_file), "-t", "3",
         "-vf", "scale=1000:420", "-pix_fmt", "yuv420p", "-r", "5", str(path)],
        check=True,
        capture_output=True,
    )
    return path
