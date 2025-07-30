import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


def find_clips() -> list[Path]:
    """Return a list of highlight clip paths."""
    highlight_dir = Path('highlights')
    clips = sorted(highlight_dir.glob('**/*.mp4'))
    if not clips:
        clips = sorted(Path('video/manual_uploads').glob('*.MP4'))[:3]
    return clips


def create_coaches_cut(clips: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    list_path = out_dir / 'highlights_list.txt'
    with list_path.open('w') as f:
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")
    output_mp4 = out_dir / 'coaches_cut.mp4'
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', str(list_path),
        '-c', 'copy', str(output_mp4)
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[!] ffmpeg not found. Skipping video concatenation.")
    return list_path


def generate_summary(out_dir: Path) -> Path:
    summary_files = sorted(Path('output/summary').glob('*_summary.json'))
    play_counter: Counter[str] = Counter()
    total_plays = 0
    for path in summary_files:
        with path.open() as f:
            data = json.load(f)
        play_counter.update(data.get('play_counts', {}))
        total_plays += data.get('total_plays', 0)
    lines = [
        f"MCA Game Analysis Summary - {datetime.now().date()}",
        '',
        f"Total clips analyzed: {len(summary_files)}",
        f"Total plays: {total_plays}",
        '',
        'Play Type Counts:'
    ]
    for play, count in play_counter.most_common():
        lines.append(f"- {play}: {count}")
    text_path = out_dir / 'analysis_summary.txt'
    text_path.write_text('\n'.join(lines) + '\n')
    return text_path


def main() -> None:
    out_dir = Path('analysis')
    clips = find_clips()
    if clips:
        create_coaches_cut(clips, out_dir)
    generate_summary(out_dir)
    print(f'Coaches cut and summary saved to {out_dir}/')


if __name__ == '__main__':
    main()
