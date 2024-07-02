import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


class YoutubeScraper:
    def __init__(self, youtube_api_key):
        self.youtube_api_key = youtube_api_key
        self.youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    def get_channel_id(self, handle: str) -> str:
        request = self.youtube.search().list(
            part='snippet',
            q=handle,
            type='channel'
        )
        response = request.execute()
        for item in response['items']:
            return item['id']['channelId']
        return ""

    def get_uploaded_videos(self, channel_id, published_after):
        videos = []
        request = self.youtube.search().list(
            part='snippet',
            channelId=channel_id,
            publishedAfter=published_after,
            type='video',
            order='date',
            maxResults=50
        )

        while request:
            response = request.execute()
            for item in response['items']:
                video_published_at = datetime.strptime(item['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                if video_published_at > datetime.utcnow() - timedelta(hours=48):
                    video = {
                        'videoId': item['id']['videoId'],
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'publishedAt': item['snippet']['publishedAt'],
                        'channelId': item['snippet']['channelId'],
                        'channelTitle': item['snippet']['channelTitle']
                    }
                    videos.append(video)
            request = self.youtube.search().list_next(request, response)

        return videos

    def get_videos_for_channels(self, channel_ids, hours=48):
        published_after = (datetime.utcnow() - timedelta(hours=hours)).isoformat("T") + "Z"
        all_videos = []

        for channel_id in channel_ids:
            channel_videos = self.get_uploaded_videos(channel_id, published_after)
            all_videos.extend(channel_videos)

        return all_videos

    def get_transcript(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([entry['text'] for entry in transcript])
        except (TranscriptsDisabled, NoTranscriptFound):
            return "Transcript not available"

    def scrape(self, channel_handles, hours=48):
        channel_ids = [self.get_channel_id(handle) for handle in channel_handles]
        videos = self.get_videos_for_channels(channel_ids, hours)

        for video in videos:
            video['transcript'] = self.get_transcript(video['videoId'])

        return pd.DataFrame(videos)

if __name__ == "__main__":
    load_dotenv()
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
    youtubeScraper = YoutubeScraper(YOUTUBE_API_KEY)
    
    # List of channel handles you want to check
    channel_handles = [
        '@MachineLearningStreetTalk',  # Example Channel Handle
        '@ArxivPapers'  # Example Channel Handle
        # Add more channel handles as needed
    ]

    # Scrape videos from the last 48 hours
    df = youtubeScraper.scrape(channel_handles, hours=48)
    print(df)

    # If you want to save to a CSV file
    df.to_csv('uploaded_videos_with_transcripts.csv', index=False)
