const routeButton = document.getElementById("routeButton");
const userText = document.getElementById("userText");

const result = document.getElementById("result");
const analysis = document.getElementById("analysis");
const error = document.getElementById("error");

const intent = document.getElementById("intent");
const confidence = document.getElementById("confidence");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const sipEndpoint = document.getElementById("sipEndpoint");

const analysisText = document.getElementById("analysisText");

const historyList = document.getElementById("historyList");
const clearHistory = document.getElementById("clearHistory");


let routingHistory = [];


// =====================================================
// VOICE INPUT
// =====================================================

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

let recognition = null;
let isListening = false;


// Create microphone button dynamically

const microphoneButton = document.createElement("button");

microphoneButton.type = "button";
microphoneButton.id = "microphoneButton";
microphoneButton.innerHTML = "🎙️ Speak";

microphoneButton.style.marginTop = "10px";
microphoneButton.style.width = "100%";
microphoneButton.style.padding = "13px";
microphoneButton.style.border = "1px solid #30364d";
microphoneButton.style.borderRadius = "10px";
microphoneButton.style.background = "#0c101c";
microphoneButton.style.color = "#9a94ff";
microphoneButton.style.cursor = "pointer";
microphoneButton.style.fontWeight = "bold";


// Put microphone button directly below textarea

userText.parentElement.insertAdjacentElement(
    "afterend",
    microphoneButton
);


// Check browser support

if (SpeechRecognition) {

    recognition = new SpeechRecognition();

    recognition.continuous = false;

    recognition.interimResults = true;

    recognition.lang = "en-IN";


    recognition.onstart = () => {

        isListening = true;

        microphoneButton.innerHTML =
            "🔴 Listening... Speak now";

        microphoneButton.style.color = "#ff7777";

    };


    recognition.onresult = (event) => {

        let transcript = "";

        for (
            let i = event.resultIndex;
            i < event.results.length;
            i++
        ) {

            transcript +=
                event.results[i][0].transcript;

        }

        userText.value = transcript;

    };


    recognition.onerror = (event) => {

        console.error(
            "Speech recognition error:",
            event.error
        );

        isListening = false;

        microphoneButton.innerHTML =
            "🎙️ Speak";

        microphoneButton.style.color =
            "#9a94ff";

        if (event.error === "not-allowed") {

            error.textContent =
                "Microphone permission was denied. Please allow microphone access in your browser.";

            error.classList.remove("hidden");

        }

    };


    recognition.onend = () => {

        isListening = false;

        microphoneButton.innerHTML =
            "🎙️ Speak";

        microphoneButton.style.color =
            "#9a94ff";

    };


    microphoneButton.addEventListener(
        "click",
        () => {

            if (isListening) {

                recognition.stop();

                return;

            }


            error.classList.add("hidden");

            userText.focus();

            recognition.start();

        }
    );

}
else {

    microphoneButton.innerHTML =
        "🎙️ Voice input not supported in this browser";

    microphoneButton.disabled = true;

    microphoneButton.style.opacity = "0.5";

}


// =====================================================
// ROUTE REQUEST
// =====================================================

routeButton.addEventListener("click", async () => {

    const text = userText.value.trim();

    if (!text) {

        userText.focus();

        return;

    }


    result.classList.add("hidden");

    error.classList.add("hidden");

    analysis.classList.remove("hidden");

    routeButton.disabled = true;


    analysisText.textContent =
        "Analyzing request...";

    await wait(500);

    analysisText.textContent =
        "Classifying intent with BERT...";

    await wait(700);

    analysisText.textContent =
        "Selecting SIP destination...";


    try {

        const response = await fetch(
            "http://127.0.0.1:8000/route",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );


        if (!response.ok) {

            throw new Error(
                "API request failed."
            );

        }


        const data =
            await response.json();


        // Display intent

        intent.textContent =
            data.intent;


        // Calculate confidence

        const percentage =
            data.confidence * 100;


        confidence.textContent =
            percentage.toFixed(2) + "%";


        confidenceText.textContent =
            getConfidenceLabel(percentage);


        confidenceBar.style.width =
            percentage + "%";


        // Display SIP endpoint

        sipEndpoint.textContent =
            data.sip_endpoint;


        analysis.classList.add("hidden");

        result.classList.remove("hidden");


        // Save request

        addToHistory({

            text: data.text,

            intent: data.intent,

            confidence: percentage,

            endpoint: data.sip_endpoint

        });

    }


    catch (err) {

        analysis.classList.add("hidden");

        error.textContent =
            "Could not connect to the BERT API. Make sure the FastAPI server is running.";

        error.classList.remove("hidden");

        console.error(err);

    }


    finally {

        routeButton.disabled = false;

    }

});


// =====================================================
// ENTER KEY
// =====================================================

userText.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            routeButton.click();

        }

    }
);


// =====================================================
// HISTORY
// =====================================================

function addToHistory(item) {

    routingHistory.unshift(item);


    if (routingHistory.length > 10) {

        routingHistory.pop();

    }


    renderHistory();

}


function renderHistory() {

    if (routingHistory.length === 0) {

        historyList.innerHTML = `
            <div class="empty-history">
                No requests yet. Your routing activity will appear here.
            </div>
        `;

        return;

    }


    historyList.innerHTML =
        routingHistory.map(item => `

            <div class="history-item">

                <div class="history-text">
                    ${escapeHTML(item.text)}
                </div>

                <div class="history-intent">
                    ${escapeHTML(item.intent)}
                    · ${item.confidence.toFixed(1)}%
                </div>

                <div class="history-endpoint">
                    ${escapeHTML(item.endpoint)}
                </div>

            </div>

        `).join("");

}


// =====================================================
// CLEAR HISTORY
// =====================================================

clearHistory.addEventListener(
    "click",
    () => {

        routingHistory = [];

        renderHistory();

    }
);


// =====================================================
// CONFIDENCE
// =====================================================

function getConfidenceLabel(value) {

    if (value >= 80) {

        return "High confidence";

    }


    if (value >= 60) {

        return "Moderate confidence";

    }


    return "Low confidence";

}


// =====================================================
// DELAY
// =====================================================

function wait(milliseconds) {

    return new Promise(
        resolve =>
            setTimeout(resolve, milliseconds)
    );

}


// =====================================================
// HTML SECURITY
// =====================================================

function escapeHTML(value) {

    return String(value)

        .replaceAll("&", "&amp;")

        .replaceAll("<", "&lt;")

        .replaceAll(">", "&gt;")

        .replaceAll('"', "&quot;")

        .replaceAll("'", "&#039;");

}


// =====================================================
// INITIALIZE
// =====================================================

renderHistory();