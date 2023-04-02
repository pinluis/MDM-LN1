document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("camera-preview");
    const canvas = document.createElement("canvas");
    const startCameraBtn = document.getElementById("start-camera");
    const captureBtn = document.getElementById("capture");
    const outputImage = document.getElementById("output-image");

    startCameraBtn.addEventListener("click", async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = stream;
            video.play();
        } catch (err) {
            console.error("Error: ", err.message, err.stack);
        }
    });

    captureBtn.addEventListener("click", () => {
        try {
            const context = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL("image/jpeg");
            sendImageToBackend(imageData);
        } catch (err) {
            console.error("Error: ", err.message, err.stack);
        }
    });

    async function sendImageToBackend(imageDataUrl) {
        const img_base64 = imageDataUrl.split(",")[1]; // Extract the base64 data without the prefix
        const response = await fetch('/detect_faces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: img_base64 }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        const imageData = result.image;
        displayOutputImage('data:image/jpeg;base64,' + imageData);
    }

    function displayOutputImage(imageSrc) {
        outputImage.src = imageSrc;
        outputImage.style.display = 'block';
    }
});
