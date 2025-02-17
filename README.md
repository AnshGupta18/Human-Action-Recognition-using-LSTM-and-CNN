!pip install pafy youtube-dl moviepy
!wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar

#Extract the Dataset
!unrar x UCF50.rar


!pip install yt-dlp
!yt-dlp -f best "https://www.youtube.com/watch?v=8u0qjmHIOcE" -o "test_videos/video.mp4"
!pip install --upgrade youtube-dl

!yt-dlp -f best "https://www.youtube.com/watch?v=fc3w827kwyA" -o "test_videos/video.mp4"
