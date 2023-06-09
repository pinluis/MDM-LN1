document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("camera-preview");
    const canvas = document.createElement("canvas");
    const startCameraBtn = document.getElementById("start-camera");
    const captureBtn = document.getElementById("capture");
    const outputImage = document.getElementById("output-image");

    startCameraBtn.addEventListener("click", async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            video.play();
        } catch (err) {
            console.error("Error: ", err.message, err.stack);
        }
    });


    captureBtn.addEventListener("click", async () => {
        try {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            await new Promise((resolve) => setTimeout(resolve, 100));

            const context = canvas.getContext("2d");
            context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
            const imageData = canvas.toDataURL("image/jpeg");
            sendImageToBackend(imageData);
        } catch (err) {
            console.error("Error: ", err.message, err.stack);
        }
    });

    async function sendImageToBackend(imageDataUrl) {
      const img_base64 = imageDataUrl.split(",")[1];
      const response = await fetch("/detect_faces", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: img_base64 }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const imageData = result.image;
      displayOutputImage("data:image/jpeg;base64," + imageData);
    }

    function displayOutputImage(imageSrc) {
        outputImage.src = imageSrc;
        outputImage.style.display = "block";
    }

  });
