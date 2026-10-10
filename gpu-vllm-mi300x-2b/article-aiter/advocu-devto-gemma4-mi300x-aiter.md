# Advocu activity — paste into app.advocu.com

Add new activity -> New activity -> Content creation -> Regular form.
Once the article is public you can instead paste the Link to Content into
"Generate your activity with AI" and check what it produces against this.

## Content type
Articles

## What was the title?
VLLM_ROCM_USE_AITER=1 Slows Gemma 4 on an AMD MI300X: What the Flag Changes and Why

## What was it about?
AMD's AITER kernel library is the usual first switch for vLLM speed on an Instinct MI300X. On Gemma 4 12B fp8 it made every one of nine cells 0.6% to 2.6% slower. The boot logs show why: Gemma 4's 512-wide attention heads keep attention on Triton, and AITER's fp8 matrix multiply has no tuned settings for any of the model's six weight shapes.

## Tags
gemma, amd, machinelearning, llm

## How many people read your content?
3000 (standing estimate, not a counter reading)

## Date published
2026-10-10

## Link to Content
https://dev.to/gde/vllmrocmuseaiter1-slows-gemma-4-on-an-amd-mi300x-what-the-flag-changes-and-why-69f

## Cover image
advocu-devto-gemma4-mi300x-aiter-cover.webp (1376x578 WebP, 2 KB, from devto-mi300x-aiter-cover.923b77d8.jpg)

---
Save as draft rather than submitting, and read it back before you do.
