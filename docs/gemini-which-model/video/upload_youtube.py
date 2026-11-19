#!/usr/bin/env python3
"""Upload the decision-flow video to YouTube with the Data API v3: video, thumbnail, captions.

  python3 upload_youtube.py --client-secret ~/.config/youtube-upload/client_secret.json --privacy unlisted
  python3 upload_youtube.py ... --dry-run          # print what would be sent, call nothing

Title, description and tags come from youtube.md. The OAuth token is cached in
~/.config/youtube-upload/token.json (mode 0600). The first run opens a browser for consent.
Quota (developers.google.com/youtube/v3/docs, read 2026-11-19): videos.insert 1 unit in the Video Uploads
bucket, thumbnails.set ~50, captions.insert 400. Writes the result to out/youtube-upload.json.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CONF = os.path.expanduser("~/.config/youtube-upload")
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.force-ssl"]


def sheet():
    t = open(os.path.join(HERE, "youtube.md")).read()
    sec = dict(re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", t, re.S | re.M))
    return {
        "title": sec["Title"].strip(),
        "description": sec["Description"].strip(),
        "tags": [x.strip() for x in sec["Tags"].split(",") if x.strip()],
    }


def creds(secret):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    tok = os.path.join(CONF, "token.json")
    c = Credentials.from_authorized_user_file(tok, SCOPES) if os.path.exists(tok) else None
    if c and c.expired and c.refresh_token:
        c.refresh(Request())
    if not c or not c.valid:
        c = InstalledAppFlow.from_client_secrets_file(secret, SCOPES).run_local_server(port=0, open_browser=True)
    os.makedirs(CONF, exist_ok=True)
    with open(os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(c.to_json())
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-secret", default=os.path.join(CONF, "client_secret.json"))
    ap.add_argument("--privacy", choices=["public", "unlisted", "private"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s = sheet()
    if len(s["title"]) > 100:
        raise SystemExit(f"title is {len(s['title'])} chars; YouTube takes 100")
    body = {
        "snippet": {
            **s,
            "categoryId": "28",
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },  # 28 = Science & Technology
        "status": {"privacyStatus": a.privacy, "selfDeclaredMadeForKids": False, "embeddable": True},
    }
    files = {
        k: os.path.join(OUT, v)
        for k, v in (("video", "decision-flow.mp4"), ("thumbnail", "thumbnail.jpg"), ("captions", "captions.srt"))
    }
    for k, p in files.items():
        if not os.path.exists(p):
            raise SystemExit(f"missing {k}: {p}")
    if a.dry_run:
        print(json.dumps({"body": body, "files": files}, indent=1))
        return

    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    yt = build("youtube", "v3", credentials=creds(a.client_secret))
    req = yt.videos().insert(
        part="snippet,status",
        body=body,
        media_body=MediaFileUpload(files["video"], mimetype="video/mp4", resumable=True, chunksize=8 << 20),
    )
    resp = None
    while resp is None:
        st, resp = req.next_chunk()
        if st:
            print(f"uploaded {st.progress():.0%}")
    vid = resp["id"]
    print("video", vid)
    result = {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "privacy": a.privacy}
    try:
        yt.thumbnails().set(
            videoId=vid, media_body=MediaFileUpload(files["thumbnail"], mimetype="image/jpeg")
        ).execute()
        result["thumbnail"] = "set"
    except Exception as e:  # custom thumbnails need a channel allowed to use them
        result["thumbnail"] = f"failed: {str(e)[:200]}"
    try:
        yt.captions().insert(
            part="snippet",
            body={"snippet": {"videoId": vid, "language": "en", "name": "English", "isDraft": False}},
            media_body=MediaFileUpload(files["captions"], mimetype="application/octet-stream"),
        ).execute()
        result["captions"] = "set"
    except Exception as e:
        result["captions"] = f"failed: {str(e)[:200]}"
    json.dump(result, open(os.path.join(OUT, "youtube-upload.json"), "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
