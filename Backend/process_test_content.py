"""
Backend Content Processor
Processes test content and sends results to API for frontend display
"""

import requests
import json
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"

# ============================================================
# TEST CASES - Your content to process
# ============================================================
test_cases = [
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1760455617_3743295366407199108_60511355591.mp4",
        "caption": "Check out this crazy fight that happened downtown! Things got really intense. #streetfight #viral",
        "title": "Street altercation caught on camera",
        "content_type": "SNIP",
        "duration": "4:32"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/6765428-uhd_2560_1440_25fps.mp4",
        "caption": "Amazing Tackle",
        "title": "Check this out",
        "content_type": "MINI",
        "duration": "12:45"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/lafdavlog/gorevideosonly_1758814254_3729526295152942151_60511355591.mp4",
        "caption": "Kill All American niggers",
        "title": "User story with concerning content",
        "content_type": "SNIP",
        "duration": "3:15"
    },
    {
        "media_path": "/Users/apple/Downloads/New Moderation/Backend/Videos/293-2937628_kate-upton-wallpapers-hd-kate-upton-4k-bikini.jpg",
        "caption": "Beautiful",  # No caption for this one
        "title": "Photo of a sunset over the mountains",
        "content_type": "SHOT",
        "duration": None
    }
    
]
# ============================================================


def process_content(test_case):
    """
    Process a single test case and send to backend API
    
    Args:
        test_case: Dictionary with media_path, caption, title, content_type, duration
        
    Returns:
        API response or error
    """
    
    media_path = test_case["media_path"]
    
    # Check if file exists
    if not Path(media_path).exists():
        print(f"❌ File not found: {media_path}")
        return None
    
    print(f"\n📹 Processing: {test_case['title']}")
    print(f"   Path: {media_path}")
    print(f"   Type: {test_case['content_type']}")
    
    try:
        # Call backend API to process
        response = requests.post(
            f"{API_BASE_URL}/api/process",
            params={
                "media_path": media_path,
                "caption": test_case.get("caption"),
                "title": test_case["title"],
                "content_type": test_case["content_type"],
                "duration": test_case.get("duration")
            },
            timeout=300  # 5 minute timeout for processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Processed successfully! Content ID: {result['content_id']}")
            
            # Show flags
            flags = result['data']['flags']
            detected = [k for k, v in flags.items() if v['detected']]
            if detected:
                print(f"⚠️  Flags detected: {', '.join(detected)}")
                print(f" Results: results = {json.dumps(flags, indent=2)}")
            else:
                print(f"✓  No issues detected")
            
            return result
        else:
            print(f"❌ Processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
        
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend API at {API_BASE_URL}")
        print(f"   Make sure the backend server is running: python api_server.py")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is healthy")
            print(f"   Models loaded: {data['loaded']}")
            print(f"   Total content: {data['total_content']}")
            return True
        else:
            print("⚠️  Backend responded but status is not healthy")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {API_BASE_URL}")
        print(f"   Start the backend with: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


def get_current_content():
    """Fetch currently moderated content from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/content")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Current Content in System: {data['count']} items")
            
            if data['count'] > 0:
                print("\n Current items:")
                for item in data['content']:
                    risk_flags = [k for k, v in item['flags'].items() if v['detected']]
                    risk_str = f"⚠️  {', '.join(risk_flags)}" if risk_flags else "✓ Clean"
                    print(f"   [{item['id']}] {item['title']} - {risk_str}")
            
            return data
        else:
            print(f"❌ Failed to fetch content: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching content: {str(e)}")
        return None


def clear_all_content():
    """Clear all content from backend"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/clear")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cleared {data['message']}")
            return True
        else:
            print(f"❌ Failed to clear content: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error clearing content: {str(e)}")
        return False


def main():
    """Main processing function"""
    
    print("=" * 70)
    print("Content Moderation - Backend Processor")
    print("=" * 70)
    
    # Check backend health
    print("\n1. Checking backend health...")
    if not check_backend_health():
        print("\n⚠️  Please start the backend server first:")
        print("   python api_server.py")
        return
    
    # Show current content
    print("\n2. Checking current content...")
    get_current_content()
    
    # Ask if user wants to clear
    print("\n3. Process new content?")
    choice = input("   Clear existing and process test cases? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n   Clearing existing content...")
        clear_all_content()
        
        # Process all test cases
        print("\n4. Processing test cases...")
        print("=" * 70)
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]")
            result = process_content(test_case)
            if result:
                results.append(result)
        
        print("\n" + "=" * 70)
        print(f"✅ Processing Complete!")
        print(f"   Processed: {len(results)}/{len(test_cases)} items")
        print("\n🎨 Open your frontend to view the results!")
        print(f"   Results available at: {API_BASE_URL}/api/content")
        print("=" * 70)
    else:
        print("\n   Skipped processing. Existing content retained.")
        print(f"   View at: {API_BASE_URL}/api/content")


if __name__ == "__main__":
    main()