// static/js/main.js
// This file contains additional JavaScript functionality for the application

// Speech and emotion analysis utilities
class EmotionAnalyzer {
    constructor() {
        // Pre-defined emotion patterns to detect in speech
        this.emotionPatterns = {
            happy: ['!', 'haha', 'wonderful', 'great', 'excellent', 'amazing'],
            sad: ['unfortunately', 'sadly', 'regret', 'sorry', 'miss'],
            angry: ['!', '?!', 'terrible', 'horrible', 'awful'],
            surprised: ['wow', 'oh', 'unexpected', 'surprised', '?!'],
            neutral: []
        };
    }

    // Detect emotion from text
    detectEmotion(text) {
        // Default to neutral
        let detectedEmotion = 'neutral';
        let highestScore = 0;
        
        // Convert to lowercase for comparison
        const lowercaseText = text.toLowerCase();
        
        // Check for each emotion pattern
        for (const [emotion, patterns] of Object.entries(this.emotionPatterns)) {
            let score = 0;
            
            // Count occurrences of each pattern
            for (const pattern of patterns) {
                const regex = new RegExp(pattern, 'gi');
                const matches = lowercaseText.match(regex);
                if (matches) {
                    score += matches.length;
                }
            }
            
            // Check for exclamation marks (intensity)
            if (emotion === 'happy' || emotion === 'angry') {
                const exclamations = (lowercaseText.match(/!/g) || []).length;
                score += exclamations;
            }
            
            // Update detected emotion if score is higher
            if (score > highestScore) {
                highestScore = score;
                detectedEmotion = emotion;
            }
        }
        
        return detectedEmotion;
    }
    
    // Adjust TTS parameters based on emotion
    getTtsParams(emotion) {
        const params = {
            rate: 1.0,  // Speech rate (default: 1.0)
            pitch: 1.0, // Voice pitch (default: 1.0)
            emphasis: 1.0 // Emphasis level (default: 1.0)
        };
        
        // Adjust parameters based on emotion
        switch (emotion) {
            case 'happy':
                params.rate = 1.1;
                params.pitch = 1.15;
                params.emphasis = 1.2;
                break;
            case 'sad':
                params.rate = 0.9;
                params.pitch = 0.9;
                params.emphasis = 0.8;
                break;
            case 'angry':
                params.rate = 1.1;
                params.pitch = 1.05;
                params.emphasis = 1.4;
                break;
            case 'surprised':
                params.rate = 1.15;
                params.pitch = 1.2;
                params.emphasis = 1.3;
                break;
            // neutral uses default values
        }
        
        return params;
    }
}

// Speaker detection and voice synchronization
class SpeakerManager {
    constructor() {
        this.speakers = new Map();
        this.nextSpeakerId = 0;
    }
    
    // Register a new speaker with face embedding
    registerSpeaker(faceEmbedding, voiceEmbedding) {
        const speakerId = this.nextSpeakerId++;
        this.speakers.set(speakerId, {
            faceEmbedding,
            voiceEmbedding,
            voiceCharacteristics: this.analyzeVoice(voiceEmbedding)
        });
        return speakerId;
    }
    
    // Analyze voice characteristics (pitch, speed, etc.)
    analyzeVoice(voiceEmbedding) {
        // In a real implementation, this would analyze actual voice features
        // For now, we'll return placeholder values
        return {
            basePitch: Math.random() * 0.5 + 0.75, // 0.75 to 1.25
            speed: Math.random() * 0.3 + 0.85,     // 0.85 to 1.15
            timbre: Math.random(),                 // 0 to 1
            gender: Math.random() > 0.5 ? 'male' : 'female'
        };
    }
    
    // Match a face embedding to existing speakers
    matchFaceToSpeaker(faceEmbedding) {
        let bestMatch = null;
        let highestSimilarity = 0;
        
        for (const [speakerId, data] of this.speakers.entries()) {
            const similarity = this.calculateSimilarity(faceEmbedding, data.faceEmbedding);
            if (similarity > 0.8 && similarity > highestSimilarity) {
                highestSimilarity = similarity;
                bestMatch = speakerId;
            }
        }
        
        return bestMatch;
    }
    
    // Calculate similarity between embeddings (cosine similarity)
    calculateSimilarity(embedding1, embedding2) {
        // In a real implementation, this would calculate actual cosine similarity
        // For now, return a random value for demonstration
        return 0.85 + Math.random() * 0.15; // 0.85 to 1.0 (high similarity for demo)
    }
    
    // Get voice parameters for a speaker to maintain consistency
    getVoiceParams(speakerId) {
        if (!this.speakers.has(speakerId)) {
            return null;
        }
        
        const speaker = this.speakers.get(speakerId);
        
        return {
            basePitch: speaker.voiceCharacteristics.basePitch,
            speed: speaker.voiceCharacteristics.speed,
            timbre: speaker.voiceCharacteristics.timbre,
            preferredVoice: speaker.voiceCharacteristics.gender === 'male' ? 
                ['p326', 'p330'] : ['p339', 'p362']
        };
    }
}

// Video processing utilities
class VideoProcessor {
    constructor() {
        this.emotionAnalyzer = new EmotionAnalyzer();
        this.speakerManager = new SpeakerManager();
    }
    
    // Process video segments with enhanced synchronization
    async processSegments(segments, targetLanguage) {
        const processedSegments = [];
        
        for (const segment of segments) {
            // Detect emotion in the text
            const emotion = this.emotionAnalyzer.detectEmotion(segment.text);
            const emotionParams = this.emotionAnalyzer.getTtsParams(emotion);
            
            // Get speaker-specific voice parameters
            let voiceParams = null;
            if (segment.speakerId !== undefined) {
                voiceParams = this.speakerManager.getVoiceParams(segment.speakerId);
            }
            
            // Combine emotion and speaker parameters
            const combinedParams = this.combineParams(emotionParams, voiceParams);
            
            // Translate with emotion preservation
            const translatedText = await this.translateWithEmotion(
                segment.text, 
                targetLanguage, 
                emotion
            );
            
            // Process timing for lip-sync
            const timing = this.calculatePreciseTiming(
                segment.start,
                segment.end,
                segment.text.length,
                translatedText.length
            );
            
            processedSegments.push({
                ...segment,
                translatedText,
                emotion,
                voiceParams: combinedParams,
                timing
            });
        }
        
        return processedSegments;
    }
    
    // Combine emotion and speaker parameters
    combineParams(emotionParams, speakerParams) {
        if (!speakerParams) {
            return emotionParams;
        }
        
        return {
            rate: emotionParams.rate * speakerParams.speed,
            pitch: emotionParams.pitch * speakerParams.basePitch,
            emphasis: emotionParams.emphasis,
            preferredVoice: speakerParams.preferredVoice
        };
    }
    
    // Translate text while preserving emotional markers
    async translateWithEmotion(text, targetLanguage, emotion) {
        // This would call the actual backend API
        // For now, we'll return a placeholder
        return `[${emotion}] ${text} (translated to ${targetLanguage})`;
    }
    
    // Calculate precise timing for lip-sync
    calculatePreciseTiming(startTime, endTime, originalLength, translatedLength) {
        const originalDuration = endTime - startTime;
        
        // Estimate duration based on character count ratio
        const ratio = translatedLength / originalLength;
        const estimatedDuration = originalDuration * ratio;
        
        // Adjust for typical speaking rate differences between languages
        const languageFactors = {
            'en_XX': 1.0,
            'es_XX': 1.1,  // Spanish typically faster than English
            'fr_XX': 1.15, // French typically faster
            'de_DE': 0.95, // German typically slower
            'zh_CN': 0.85, // Mandarin typically slower
            'ja_XX': 0.9,  // Japanese typically slower
            'ru_RU': 0.9   // Russian typically slower
            // Add other languages as needed
        };
        
        const langFactor = languageFactors[targetLanguage] || 1.0;
        const adjustedDuration = estimatedDuration * langFactor;
        
        // Calculate new end time, ensuring it doesn't overlap with next segment
        const newEndTime = startTime + adjustedDuration;
        
        return {
            start: startTime,
            end: newEndTime,
            needsStretching: adjustedDuration > originalDuration * 1.2 || adjustedDuration < originalDuration * 0.8
        };
    }
}

// Initialize the video processor when the page loads
document.addEventListener('DOMContentLoaded', function() {
    window.videoProcessor = new VideoProcessor();
    console.log('Advanced video processing functionality initialized');
});