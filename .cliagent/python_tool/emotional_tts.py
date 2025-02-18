import typing
from typing import List, Dict

def express_emotions() -> Dict[str, str]:
    emotions: Dict[str, str] = {
        "happy": "Wow! I'm so excited and thrilled to help you today! 🎉",
        "sad": "Oh no... I feel really down and blue about that... *sigh*",
        "angry": "Grrrr! This is absolutely frustrating and unacceptable!",
        "surprised": "Oh my goodness! I can't believe what I'm seeing! 😮",
        "scared": "Eek! This is terrifying... I'm shaking!",
        "calm": "Everything is peaceful and tranquil... just breathe...",
        "love": "I absolutely adore this! It fills my heart with joy! ❤️"
    }
    
    return emotions

def main() -> None:
    emotion_dict = express_emotions()
    for emotion, expression in emotion_dict.items():
        print(f"{emotion.upper()}: {expression}\n")

if __name__ == "__main__":
    main()
