# System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                    React Dashboard (Port 3000)                       │
│                                                                      │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Upload   │  │   Content    │  │    Review    │               │
│  │   Button   │  │     List     │  │   Controls   │               │
│  └────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  Components: ContentModerationDashboard.jsx                         │
└────────────┬─────────────────────────────────────────────────────┘
             │
             │ HTTP Requests (POST /api/moderate)
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API SERVICE LAYER                               │
│                      api_service.js                                  │
│                                                                      │
│  • transformModerationResults()                                     │
│  • moderateContent()                                                │
│  • getModerationQueue()                                             │
│  • approveContent() / rejectContent()                               │
└────────────┬─────────────────────────────────────────────────────┘
             │
             │ REST API Calls
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FLASK REST API                                  │
│                    api_server.py (Port 5000)                         │
│                                                                      │
│  Endpoints:                                                          │
│  • POST   /api/moderate      ← Upload & analyze content            │
│  • GET    /api/queue         ← Get moderation queue                │
│  • POST   /api/review/<id>   ← Submit review decision              │
│  • GET    /api/stats         ← Get statistics                      │
│  • GET    /api/health        ← Health check                        │
└────────────┬─────────────────────────────────────────────────────┘
             │
             │ Python Function Calls
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               CONTENT MODERATION SYSTEM                              │
│              merged_moderation.py                                    │
│                                                                      │
│  Class: ContentModerationSystem                                     │
│  Method: moderate_content(media_path, caption)                      │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │              DETECTION PIPELINE                            │   │
│  │                                                            │   │
│  │  1. Violence Detection                                    │   │
│  │     ├─ VideoMAE (videos & images)                        │   │
│  │     └─ Output: Violence/NonViolence + confidence         │   │
│  │                                                            │   │
│  │  2. NSFW Detection                                        │   │
│  │     ├─ Images:                                            │   │
│  │     │  ├─ Level 1: Transformer (porn/sexy/neutral...)    │   │
│  │     │  ├─ Level 2: YOLOv8 (body part detection)         │   │
│  │     │  └─ Bidirectional override logic                   │   │
│  │     │                                                      │   │
│  │     └─ Videos:                                            │   │
│  │        ├─ Extract 10 frames                              │   │
│  │        ├─ Level 1 on all frames                          │   │
│  │        └─ Level 2 on flagged frames only                 │   │
│  │                                                            │   │
│  │  3. Hate Speech Detection                                │   │
│  │     ├─ Primary: acceptable/inappropriate/offensive/violent│   │
│  │     └─ Subcategory (if flagged):                         │   │
│  │        └─ community/racial_ethnic/religious/nationality   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Dependencies:                                                       │
│  ├─ VideoMAEViolenceDetector                                       │
│  ├─ NSFWDetector (Level 1 + Level 2)                              │
│  └─ HateSpeechDetector (Primary + Subcategory)                    │
└────────────┬─────────────────────────────────────────────────────┘
             │
             │ Model Inference
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AI MODELS                                     │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │    VideoMAE      │  │  NSFW Level 1    │  │  NSFW Level 2   │ │
│  │   (Violence)     │  │  (Transformer)   │  │   (YOLOv8)      │ │
│  │                  │  │                  │  │                 │ │
│  │ • Input: 16      │  │ • Input: Image   │  │ • Input: Image  │ │
│  │   frames         │  │ • Output: 5      │  │ • Output: Body  │ │
│  │ • Output:        │  │   categories     │  │   part boxes    │ │
│  │   Violence class │  │ • Fast screening │  │ • 6 NSFW classes│ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐                        │
│  │  Hate Speech     │  │  Subcategory     │                        │
│  │   Primary        │  │   Classifier     │                        │
│  │                  │  │                  │                        │
│  │ • Input: Text    │  │ • Input: Text    │                        │
│  │ • Output: 4      │  │ • Output: 5      │                        │
│  │   severity levels│  │   hate types     │                        │
│  │ • Always runs    │  │ • Conditional    │                        │
│  └──────────────────┘  └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘


DATA FLOW EXAMPLE:
═════════════════

Upload Video with Caption
         │
         ▼
┌────────────────────┐
│ Frontend uploads:  │
│ • video.mp4        │
│ • "Test caption"   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Flask API receives │
│ Saves to /tmp      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────────────────────┐
│ Moderation System analyzes:        │
│                                    │
│ 1. Violence: 10 frames → Violence  │
│    Confidence: 89%                 │
│                                    │
│ 2. NSFW: 10 frames                │
│    Frame 3 flagged → Level 2       │
│    Detected: FEMALE_BREAST_EXPOSED │
│    Final: NSFW                     │
│                                    │
│ 3. Text: "Test caption"            │
│    Primary: acceptable             │
│    (No subcategory needed)         │
└─────────┬──────────────────────────┘
          │
          ▼
┌────────────────────────────────────┐
│ API returns JSON:                  │
│ {                                  │
│   violence: {detected: true, ...}, │
│   nsfw: {final_rating: "nsfw",...},│
│   text: {label_name: "acceptable"} │
│ }                                  │
└─────────┬──────────────────────────┘
          │
          ▼
┌────────────────────────────────────┐
│ Frontend transforms data:          │
│                                    │
│ flags: {                           │
│   violence: {detected: true, 0.89},│
│   nudity: {detected: true, 0.87},  │
│   hateSpeech: {detected: false}    │
│ }                                  │
│                                    │
│ riskLevel: "HIGH"                  │
└─────────┬──────────────────────────┘
          │
          ▼
┌────────────────────────────────────┐
│ Dashboard displays:                │
│                                    │
│ [HIGH RISK]                        │
│ • Violence: ⚠️ DETECTED (89%)      │
│ • NSFW: ⚠️ DETECTED (87%)          │
│ • Hate Speech: ✓ CLEAN (0%)       │
│                                    │
│ [Approve] [Deny]                   │
└────────────────────────────────────┘


TECHNOLOGY STACK:
═════════════════

Frontend:
├─ React 18
├─ Tailwind CSS
├─ Lucide Icons
└─ Axios

Backend:
├─ Flask + Flask-CORS
├─ PyTorch 2.0+
├─ Transformers 4.30+
├─ OpenCV
├─ Ultralytics YOLOv8
└─ NumPy

Models:
├─ VideoMAE (HuggingFace)
├─ NSFW Transformer
├─ YOLOv8 NudeNet
├─ BERT Hate Speech
└─ Fine-tuned Subcategory


DEPLOYMENT OPTIONS:
══════════════════

Development:
├─ Backend: python api_server.py
└─ Frontend: npm start

Production:
├─ Backend: Gunicorn + Nginx
├─ Frontend: npm run build → S3/Netlify
├─ Database: PostgreSQL (queue)
└─ Storage: AWS S3 (uploads)
```
