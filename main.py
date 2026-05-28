import google.cloud.dialogflow_v2 as dialogflow
import google.cloud.speech as speech
from flask import jsonify, make_response
import base64
import os
import traceback
# The previous attempts using google.protobuf.json_format.MessageToDict failed for MapComposite.
# We will use manual recursive conversion instead.

# --- [DO NOT CHANGE] ---
# Helper function for gender string
def get_gender_string(ssml_gender):
    if ssml_gender == dialogflow.SsmlVoiceGender.SSML_VOICE_GENDER_MALE:
        return "MALE"
    elif ssml_gender == dialogflow.SsmlVoiceGender.SSML_VOICE_GENDER_FEMALE:
        return "FEMALE"
    elif ssml_gender == dialogflow.SsmlVoiceGender.SSML_VOICE_GENDER_NEUTRAL:
        return "NEUTRAL"
    return "UNSPECIFIED"

# ⭐ CRITICAL FIX: NEW HELPER FUNCTION to correctly unpack Dialogflow Value objects
def _convert_value(value):
    """Recursively converts Dialogflow Value and Struct/List objects to primitive Python types."""
    
    # Check for the underlying primitive type by checking its attribute
    if hasattr(value, 'string_value'):
        return value.string_value
    elif hasattr(value, 'number_value'):
        return value.number_value
    elif hasattr(value, 'bool_value'):
        return value.bool_value
    elif hasattr(value, 'list_value') and value.list_value:
        # Recursively convert list elements
        return [_convert_value(v) for v in value.list_value.values]
    elif hasattr(value, 'struct_value') and value.struct_value:
        # Recursively convert struct (MapComposite fields)
        return {k: _convert_value(v) for k, v in value.struct_value.fields.items()}
    elif hasattr(value, 'null_value') or value is None:
        return None
    
    # Fallback for simple Python types that may be directly in the MapComposite
    return value
# --- [END OF HELPER] ---

# Set PROJECT_ID to your AGENT'S project
PROJECT_ID = "prashanti-college-bot-seru" # Make sure this is correct

# --- API Client Initialization ---
try:
    # We specify the audio encoding we want from Dialogflow
    output_audio_config = dialogflow.OutputAudioConfig(
        audio_encoding=dialogflow.OutputAudioEncoding.OUTPUT_AUDIO_ENCODING_LINEAR_16
    )
    # Initialize Google Cloud clients
    dialogflow_client = dialogflow.SessionsClient()
    speech_client = speech.SpeechClient()
except Exception as e:
    # Log critical error if clients fail to initialize
    print(f"CRITICAL ERROR: Failed to initialize API clients: {e}")
    dialogflow_client = None
    speech_client = None
# -----------------------------

def handle_query(request):
    """Handles incoming requests, performs STT and Dialogflow detection."""
    # --- [CORS handling for OPTIONS preflight requests] ---
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    # Standard CORS header for actual POST requests
    headers = { 'Access-Control-Allow-Origin': '*' }
    # ----------------------------------------------------

    # Check if clients initialized correctly during startup
    if not dialogflow_client or not speech_client:
         print("ERROR: API clients not initialized during startup.")
         return (jsonify({"error": "Server configuration error"}), 500, headers)

    try:
        # --- Request Parsing and Validation ---
        request_json = request.get_json(silent=True)
        if not request_json:
            print("ERROR: No JSON payload received.")
            return (jsonify({"error": "No JSON payload"}), 400, headers)

        audio_base64 = request_json.get('audio_data')
        session_id = request_json.get('session_id')
        lang_code = request_json.get('lang', 'en-US') # Default to English if not provided

        if not audio_base64 or not session_id:
            print(f"ERROR: Missing 'audio_data' or 'session_id' in JSON payload.")
            return (jsonify({"error": "Missing required fields: 'audio_data' or 'session_id'"}), 400, headers)

        print(f"Received request for session: {session_id}, lang: {lang_code}")

        # --- Base64 Decoding ---
        try:
            # Decode the base64 audio data
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as decode_error:
            # Log the specific decode error and return 400
            print(f"ERROR: Failed to decode base64 audio data: {decode_error}")
            return (jsonify({"error": "Invalid audio data encoding"}), 400, headers)

        if not audio_bytes:
             print("ERROR: Decoded audio data is empty.")
             return (jsonify({"error": "Empty audio data after decoding"}), 400, headers)

        # --- 1. Speech-to-Text (STT) ---
        audio_input = speech.RecognitionAudio(content=audio_bytes)
        stt_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000, # Ensure this matches your Android app's sample rate
            language_code=lang_code
        )

        print(f"Sending audio to STT API with lang: {lang_code}...")
        try:
            stt_response = speech_client.recognize(config=stt_config, audio=audio_input)
        except Exception as stt_error:
            print(f"ERROR: Speech-to-Text API call failed: {stt_error}\n{traceback.format_exc()}")
            return (jsonify({"error": "Speech-to-Text service failed"}), 500, headers)

        transcript = ""
        if stt_response.results and stt_response.results[0].alternatives:
            transcript = stt_response.results[0].alternatives[0].transcript
            print(f"Transcript: {transcript}")
        else:
            # Handle case where STT returns no results gracefully
            print("WARNING: STT returned no transcription results.")
            bot_reply = "Sorry, I couldn't understand the audio."
            if "hi" in lang_code:
                 bot_reply = "क्षमा करें, मुझे ऑडियो समझ में नहीं आया।" # Hindi fallback
            # Return 200 OK but indicate no transcript/understanding
            # ***** MODIFICATION 1 *****
            # Added empty intent/parameters so the app doesn't crash
            return (jsonify({
                "response_text": bot_reply,
                "transcript": "",
                "lang_code_sent": lang_code,
                "intent_name": "stt.no_result",
                "parameters": {}
            }), 200, headers)

        # --- 2. Dialogflow Detect Intent ---
        session_path = dialogflow_client.session_path(PROJECT_ID, session_id)
        text_input = dialogflow.types.TextInput(text=transcript, language_code=lang_code)
        query_input = dialogflow.types.QueryInput(text=text_input)

        # Build the full DetectIntentRequest object explicitly
        api_request = dialogflow.DetectIntentRequest(
            session=session_path,
            query_input=query_input,
            output_audio_config=output_audio_config # Include audio config here
        )

        print(f"Sending text to Dialogflow (lang={lang_code}): '{transcript}'")

        try:
            # Call the detect_intent method using the constructed 'request' object
            df_response = dialogflow_client.detect_intent(request=api_request)
        except Exception as df_error:
             print(f"ERROR: Dialogflow API call failed: {df_error}\n{traceback.format_exc()}")
             return (jsonify({"error": "Dialogflow service failed"}), 500, headers)


        print("Received response from Dialogflow.")

        # --- Prepare Final JSON Response for Android App ---

        # ***** MODIFICATION 2: Get all the data from the query result *****
        query_result = df_response.query_result
        bot_reply = query_result.fulfillment_text
        intent_name = query_result.intent.display_name

        # ✅ FINAL FIX for MapComposite/AttributeError
        # Use the helper function to safely convert the MapComposite and its nested Value objects
        parameters_dict = {}
        if query_result.parameters:
            converted_params = {}
            for key, value in query_result.parameters.items():
                converted_params[key] = _convert_value(value)
            
            parameters_dict = converted_params
        # END FIX

        print(f"Dialogflow Reply: {bot_reply}")
        print(f"Intent: {intent_name}, Parameters: {parameters_dict}")

        # ***** MODIFICATION 3: Add the new fields to the final JSON *****
        final_response = {
            "response_text": bot_reply,
            "transcript": transcript,
            "lang_code_sent": lang_code,

            # --- NEW FIELDS FOR ANDROID APP ---
            "intent_name": intent_name,
            "parameters": parameters_dict # This is now guaranteed to be JSON-safe
            # ------------------------------------
        }

        # --- Check for voice config and add it if present ---
        if df_response.output_audio_config and df_response.output_audio_config.synthesize_speech_config:
            print("Found output_audio_config in Dialogflow response. Adding to JSON.")
            tts_config = df_response.output_audio_config.synthesize_speech_config
            voice_config = tts_config.voice

            final_response["output_audio_config"] = {
                "language_code": voice_config.language_code,
                "name": voice_config.name,
                "ssml_gender": get_gender_string(voice_config.ssml_gender)
            }
        else:
            print("No output_audio_config found in Dialogflow response.")

        # --- Send successful response back to the app ---
        # This line will now succeed as final_response contains only JSON-safe types.
        return (jsonify(final_response), 200, headers)

    # --- Generic Exception Handling for Unexpected Errors ---
    except Exception as e:
        # Log the full traceback for any unhandled errors
        print(f"ERROR: Unhandled exception in handle_query: {e}\n{traceback.format_exc()}")
        # Return a generic 500 Internal Server Error
        return (jsonify({"error": "An internal server error occurred."}), 500, headers)
