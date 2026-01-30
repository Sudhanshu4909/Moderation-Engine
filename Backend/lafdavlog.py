import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def instagram_login(driver, username, password):
    """Login to Instagram."""
    driver.get("https://www.instagram.com/accounts/login/")
    time.sleep(5)

    # Cookie consent if appears
    try:
        cookie_btn = driver.find_element(By.XPATH, "//button[contains(text(),'Allow all cookies')]")
        cookie_btn.click()
        time.sleep(2)
    except:
        pass

    # Fill credentials
    driver.find_element(By.NAME, "username").send_keys(username)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.NAME, "password").send_keys(Keys.RETURN)
    time.sleep(8)
    print("✅ Logged into Instagram successfully!")


def download_video(video_url, save_path):
    """Download mp4 file."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(video_url, stream=True, headers=headers)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"📥 Saved: {save_path}")
        else:
            print(f"⚠️ HTTP {r.status_code} for {video_url}")
    except Exception as e:
        print(f"❌ Error: {e}")


def scrape_reels_and_videos(username, password, target_account, max_scrolls=5):
    """Scrape all reels/videos from target profile."""
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument("--headless=new")  # uncomment to run in background

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
        instagram_login(driver, username, password)

        profile_url = f"https://www.instagram.com/{target_account}/"
        driver.get(profile_url)
        time.sleep(5)

        print(f"🔎 Scanning @{target_account} posts...")

        # Scroll and load posts
        for _ in range(max_scrolls):
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(3)

        # Collect post URLs
        anchors = driver.find_elements(By.TAG_NAME, "a")
        post_links = [a.get_attribute("href") for a in anchors if a.get_attribute("href") and ("/reel/" in a.get_attribute("href") or "/p/" in a.get_attribute("href"))]
        post_links = list(set(post_links))
        print(f"🧩 Found {len(post_links)} posts.\n")

        output_dir = os.path.join(os.getcwd(), target_account)
        os.makedirs(output_dir, exist_ok=True)

        for idx, link in enumerate(post_links, 1):
            print(f"▶️ Opening post {idx}/{len(post_links)}")
            driver.get(link)
            time.sleep(5)

            video_url = None

            # 1️⃣ Try to wait for the <video> tag
            try:
                video_el = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "video"))
                )
                video_url = video_el.get_attribute("src")
            except:
                pass

            # 2️⃣ If still blob, try <meta property='og:video'>
            if not video_url or video_url.startswith("blob:"):
                try:
                    meta_el = driver.find_element(By.XPATH, "//meta[@property='og:video']")
                    video_url = meta_el.get_attribute("content")
                except:
                    pass

            # 3️⃣ Optional: trigger playback to force src to appear
            if (not video_url or video_url.startswith("blob:")):
                try:
                    driver.execute_script("document.querySelector('video').play()")
                    time.sleep(4)
                    video_el = driver.find_element(By.TAG_NAME, "video")
                    video_url = video_el.get_attribute("src")
                except:
                    pass

            if video_url and video_url.startswith("https"):
                save_path = os.path.join(output_dir, f"video_{idx}.mp4")
                download_video(video_url, save_path)
            else:
                print(f"⏩ Skipped (no downloadable URL): {link}")

        print("\n🎉 All videos downloaded successfully!")

    except Exception as e:
        print(f"⚠️ Error: {e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    username = input("Enter your Instagram username: ").strip()
    password = input("Enter your Instagram password: ").strip()
    target_account = "lafdavlog"
    scrape_reels_and_videos(username, password, target_account)
