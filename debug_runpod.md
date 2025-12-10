# Debugging RunPod Serverless Handler

## Understanding the Hanging Issue

Since your handler only runs on RunPod, we need to debug it within that environment. The hanging likely occurs during:

1. **Model loading** - Missing HF_TOKEN or network issues
2. **S3 configuration** - Missing/incorrect S3 credentials
3. **Request processing** - Issues in the synthesis pipeline
4. **RunPod worker startup** - Problems with the serverless framework

## Enhanced Debugging Strategy

### 1. Environment Variable Validation

The enhanced handler now includes comprehensive validation at startup. Make sure these are set in your RunPod endpoint:

**Required Variables:**
```
HF_TOKEN=your_huggingface_token
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_ACCESS_KEY_ID=your_s3_key
S3_SECRET_ACCESS_KEY=your_s3_secret
S3_BUCKET_NAME=your_bucket_name
```

**Optional Variables:**
```
AUDIO_VOICES_DIR=/runpod-volume/echo-tts/audio_voices
OUTPUT_AUDIO_DIR=/runpod-volume/echo-tts/output_audio
S3_REGION=us-east-1
```

### 2. Request to Debug the Handler

Since you can't run the handler locally, here's how to debug on RunPod:

#### A. Test Basic Handler Response
```bash
# Use RunPod API to send a simple health check
ENDPOINT_ID="your-endpoint-id"
RUNPOD_API_KEY="your-api-key"

curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
    "input": {
      "action": "health_check"
    }
  }'
```

#### B. Test Minimal Synthesis Request
```bash
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
    "input": {
      "text": "Hello world",
      "parameters": {
        "num_steps": 10,
        "sequence_length": 160,
        "seed": 42
      }
    }
  }'
```

### 3. Checking RunPod Logs

The enhanced handler will now provide much more detailed logs. To view them:

1. **In RunPod Console**:
   - Go to your endpoint
   - Click on the worker/pod that's running
   - View the logs in real-time

2. **What to Look For**:
   - `=== Echo-TTS RunPod Handler Starting ===` - Initial startup
   - `=== HANDLER CALLED ===` - When request is received
   - `Starting model loading...` - Model loading progress
   - Any ERROR or WARNING messages

### 4. Common Issues & Solutions

#### Issue 1: Missing HF_TOKEN
**Logs you'll see:**
```
Configuration validation failed:
  - HF_TOKEN is required but not set
S3 configuration missing: ...
```

**Solution:**
Add HF_TOKEN to your RunPod endpoint environment variables.

#### Issue 2: S3 Configuration Missing
**Logs you'll see:**
```
S3 configuration missing: S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME
```

**Solution:**
Add all S3 environment variables to your RunPod endpoint.

#### Issue 3: Model Loading Hanging
**Logs you'll see:**
```
Starting model loading on device: cuda
Loading EchoDiT model from HuggingFace...
[Then it hangs here]
```

**This could be:**
- Network connectivity issues
- HuggingFace rate limiting
- Model download timeouts

#### Issue 4: Request Not Reaching Handler
**If you see no logs at all when sending requests:**
- Check if your endpoint is active
- Verify your API key and endpoint ID
- Make sure you're using the correct RunPod API URL

### 5. Progressive Debugging

#### Step 1: Verify Container Starts
1. Push your code to GitHub
2. Wait for RunPod to build the container
3. Check if the endpoint becomes "Ready"
4. Look for startup logs

#### Step 2: Test Health Check
Send a health check request first - this tests configuration without loading models.

#### Step 3: Test Minimal Request
Use the minimal synthesis request above with very short generation parameters.

#### Step 4: Full Request
Only after basic requests work, try full-length requests.

### 6. If Still Hanging

If the handler is still hanging with enhanced logging, try adding more debugging to identify the exact hanging point:

**Option A: Add More Timeouts**
The enhanced handler doesn't have timeouts on model loading. This could be the issue if HuggingFace downloads are slow.

**Option B: Pre-download Models**
Modify your bootstrap.sh to pre-download models during container build instead of at runtime.

**Option C: Use a Test Handler**
Create a minimal test handler that just returns a response without any heavy processing:

```python
def test_handler(job):
    return {"output": "Handler is working", "timestamp": time.time()}
```

Replace the main handler temporarily to isolate if the issue is with your code vs RunPod infrastructure.

### 7. Quick Debug Checklist

- [ ] All environment variables set in RunPod endpoint?
- [ ] HF_TOKEN is valid and has access to gated models?
- [ ] S3 credentials are correct and bucket exists?
- [ ] Container builds successfully on RunPod?
- [ ] Endpoint shows as "Ready"?
- [ ] Can see startup logs in RunPod console?
- [ ] Health check request works?
- [ ] Minimal synthesis request works?

### 8. Most Likely Causes

Based on your description, the most likely causes are:

1. **Missing HF_TOKEN** - This is required to download the gated Echo-TTS models
2. **Missing S3 Configuration** - Required for uploading generated audio
3. **Network Issues** - Problems connecting to HuggingFace or S3 from RunPod
4. **Request Not Reaching Handler** - Issues with API key or endpoint configuration

The enhanced handler should now show you exactly where the problem occurs through the detailed logging.