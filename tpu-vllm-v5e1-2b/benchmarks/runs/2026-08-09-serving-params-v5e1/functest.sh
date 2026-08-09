#!/usr/bin/env bash
U=http://localhost:8000/v1/chat/completions
M=google/gemma-4-E2B-it
pass(){ echo "  ✅ $1"; }; fail(){ echo "  ❌ $1 -- $2"; }

echo "[1] basic chat"
R=$(curl -s -m 120 $U -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"max_tokens\":16}")
echo "$R" | grep -q content && pass "chat: $(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'][:60])" 2>/dev/null)" || fail chat "$(echo $R | head -c 200)"

echo "[2] tool calling (the agent claim)"
R=$(curl -s -m 120 $U -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\",\"max_tokens\":128}")
echo "$R" | grep -q "tool_calls" && pass "tool_calls: $(echo "$R" | python3 -c "import json,sys;d=json.load(sys.stdin)['choices'][0]['message'];print(json.dumps(d.get('tool_calls'))[:160])" 2>/dev/null)" || fail tool "$(echo $R | head -c 250)"

echo "[3] multimodal image (validates the 2496 mm floor)"
python3 -c "
import base64,struct,zlib
def png(w,h):
    raw=b\"\".join(b\"\x00\"+bytes([(x*7)%256,(y*5)%256,128]*1) for y in range(h) for x in range(0,w*0+1)) 
    return None
" 2>/dev/null
IMG=$(python3 - <<'PY'
import base64,zlib,struct
w=h=64
raw=b"".join(b"\x00"+bytes([(x*4)%256,(y*4)%256,180]*1 for x in range(w)) if False else b"\x00"+b"".join(bytes([(x*4)%256,(y*4)%256,180]) for x in range(w)) for y in range(h))
def chunk(t,d):
    c=struct.pack(">I",len(d))+t+d
    return c+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
png=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b"")
print(base64.b64encode(png).decode())
PY
)
R=$(curl -s -m 180 $U -H "Content-Type: application/json" -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image in one short sentence.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG\"}}]}],\"max_tokens\":48}")
echo "$R" | grep -q content && pass "image: $(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'][:80])" 2>/dev/null)" || fail image "$(echo $R | head -c 250)"

echo "[4] long context (validates --max-model-len 32768)"
LONG=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. '*2600)[:120000])")
R=$(curl -s -m 300 $U -H "Content-Type: application/json" -d "$(python3 -c "
import json,sys
long=('The quick brown fox jumps over the lazy dog. '*2600)
print(json.dumps({'model':'$M','messages':[{'role':'user','content':long+' Reply with the word DONE.'}],'max_tokens':16}))")")
echo "$R" | grep -q content && pass "32k ctx ok ($(echo "$R" | python3 -c "import json,sys;print(json.load(sys.stdin)['usage']['prompt_tokens'])" 2>/dev/null) prompt tokens)" || fail longctx "$(echo $R | head -c 250)"
