import os
from collections import defaultdict
import argparse

from label_studio_sdk.client import LabelStudio

from src.label_studio.generate_png import generate_leaderboard, generate_progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate retro pixelated images.")
    parser.add_argument("--font", default="src/label_studio/lilliput_steps.otf", help="Path to a pixelated TTF font")
    parser.add_argument("--output-dir", default="export", help="Directory for output images")
    parser.add_argument("--label-studio-url", default='http://localhost:8080')
    args = parser.parse_args()
    # Connect to the Label Studio API and check the connection
    ls = LabelStudio(base_url=args.label_studio_url, api_key=os.environ.get('API_KEY'))

    global_result = defaultdict(dict)
    leaderboard_id = defaultdict(lambda:0)
    leaderboard = {}

    for project in ls.projects.list():
        project_id = project.id
        content = ls.projects.exports.as_json(project_id)
        for entry in content:
            for entry in entry['annotations']:
                user_id = entry['completed_by']
                leaderboard_id[user_id] += 1
        title = project.title
        ratio = project.finished_task_number / project.task_number

        subtitle, subhierarchy = title.split('.')[1], ".".join(title.split('.')[2:])
        global_result[subtitle][subhierarchy] = int(ratio*100)


    for user_id in leaderboard_id:
        username = ls.users.get(user_id).username
        leaderboard[username] = leaderboard_id[user_id]
    
    progress_path = os.path.join(args.output_dir, "retro_progress.png")
    leaderboard_path = os.path.join(args.output_dir, "retro_leaderboard.png")

    generate_progress(global_result, progress_path, args.font)
    generate_leaderboard(leaderboard, leaderboard_path, args.font)