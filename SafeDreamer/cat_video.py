from moviepy.editor import VideoFileClip, concatenate_videoclips

# MP4 list for merging
file_list = ['video1.mp4', 'video2.mp4', 'video3.mp4']

paths = [
    '20230614-034349_dreamerv3_safetygym_SafetyRacecarButton1-v0_0',
    '20230614-034430_dreamerv3_safetygym_SafetyRacecarButton2-v0_0',
    '20230614-034955_dreamerv3_safetygym_SafetyRacecarPush1-v0_0',
    '20230614-035100_dreamerv3_safetygym_SafetyRacecarPush2-v0_0',
    '20230614-035311_dreamerv3_safetygym_SafetyPointGoal1-v0_0',
    '20230614-035339_dreamerv3_safetygym_SafetyPointGoal2-v0_0',
    '20230614-035655_dreamerv3_safetygym_SafetyPointButton1-v0_0',
    '20230614-181543_dreamerv3_safetygym_SafetyPointButton2-v0_0',
    '20230614-181712_dreamerv3_safetygym_SafetyPointPush1-v0_0',
    '20230614-193835_dreamerv3_safetygym_SafetyDoggoGoal1-v0_0',
]

episodes = [501, 1002, 1503, 2004, 2505, 3006, 3507, 4008, 4509, 5010, 5511, 6012, 6513, 7014, 7515, 8016, 8517, 9018, 9519]

for path in paths:
    video_list = []
    for episode in episodes:
        mp4_path = './logdir/' + path + '/' + 'groundtruth_video_far_list_'+str(episode)+'-episode-0.mp4'
        video_list.append(mp4_path)
    # create VideoFileClip list object
    video_clips = [VideoFileClip(file) for file in video_list]

    # merge
    final_clip = concatenate_videoclips(video_clips)

    # output
    final_clip.write_videofile(path+'groundtruth_video_far_list_.mp4')
