from analysis.utils import run_bash_command

def generate_video_from_images(img_folder, video_file, framerate):
    bash_command = "ffmpeg -y -r "+str(framerate)+" -f image2 -s 1920x1080 -i "+img_folder+"/img_%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p "+video_file
    return run_bash_command(bash_command)

def join_video_and_audio(video_file,audio_file, trim_audio=0):
    video_file2 = video_file+"_music.mp4"
    audio_format = "mp3"
    new_audio_file = video_file+"."+audio_format
    bash_command = "ffprobe -v 0 -show_entries format=duration -of compact=p=0:nk=1 "+video_file
    duration = float(run_bash_command(bash_command))
    bash_command = "ffmpeg -y -i "+audio_file+" -ss "+str(trim_audio)+" -t "+str(duration)+" "+new_audio_file
    run_bash_command(bash_command)
    bash_command = "ffmpeg -y -i "+video_file+" -i "+new_audio_file+" "+video_file2
    run_bash_command(bash_command)
