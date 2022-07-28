import os

# This file provides the commands to extract the images from the original videos of each episode.

# Everybody Loves Raymond (ELR)
for ep_i in range(1,17):
    episode_name = 'S09E%02d' % ep_i
    video_name = 'ELR_%s' % episode_name
    os.makedirs(video_i, exist_ok=True)
    os.system('ffmpeg -i %s.mkv -qscale:v 2 -vf "yadif=parity=auto, scale=960:540" -vsync vfr -map 0:0 %s/%s_%%08d.jpg' % (video_name, episode_name, video_name))

# Frasier
for ep_i in range(1,24):
    episode_name = 'S11E%02d' % ep_i
    video_name = 'Frasier_%s' % episode_name
    os.makedirs(video_i, exist_ok=True)
    os.system('ffmpeg -i %s.mkv -qscale:v 2 -vf "yadif=parity=auto, scale=720:540" -vsync vfr -map 0:0 %s/%s_%%08d.jpg' % (video_name, episode_name, video_name))

# Friends
for ep_i in range(1,25):
    episode_name = 'S08E%02d' % ep_i
    video_name = 'Friends_%s' % episode_name
    os.makedirs(video_i, exist_ok=True)
    os.system('ffmpeg -i %s.mkv -qscale:v 2 %s/%s_%%08d.jpg' % (video_name, episode_name, video_name)

# How I Met Your Mother (HIMYM)
for ep_i in range(1,25):
    episode_name = 'S06E%02d' % ep_i
    video_name = 'HIMYM_%s' % episode_name
    os.makedirs(video_i, exist_ok=True)
    os.system('ffmpeg -i %s.mkv -vf scale=960:540 -qscale:v 2 -vsync vfr -map 0:0 %s/%s_%%08d.jpg' % (video_name, episode_name, video_name))

# Seinfeld
episodes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23]
for ep_i in episodes:
    episode_name = 'S09E%02d' % ep_i
    video_name = 'Seinfeld_%s' % episode_name
    os.makedirs(video_i, exist_ok=True)
    os.system('ffmpeg -i %s.mkv -vf scale=720:540 -qscale:v 2 -vsync vfr -map 0:0 %s/%s_%%08d.jpg' % (video_name, episode_name, video_name))

# Two And A Half Men (TAAHM)
for ep_i in range(1,24):
    episode_name = 'S10E%02d' % ep_i
    video_name = 'TAAHM_%s' % episode_name
    os.makedirs(video_i, exist_ok=True)
    os.system('ffmpeg -i %s.mkv -vf scale=960:540 -qscale:v 2 -vsync vfr -map 0:0 %s/%s_%%08d.jpg' % (video_name, episode_name, video_name))

# The Big Bang Theory (TBBT)
for ep_i in range(1,25):
    episode_name = 'S12E%02d' % ep_i
    video_name = 'TBBT_%s' % episode_name
    os.makedirs(video_i, exist_ok=True)
    os.system('ffmpeg -i %s.mkv -qscale:v 2 %s/%s_%%08d.jpg' % (video_name, episode_name, video_name))
