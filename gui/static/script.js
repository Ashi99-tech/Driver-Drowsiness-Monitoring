document.addEventListener("DOMContentLoaded", () => {
    const statusElement = document.getElementById("status");

    function checkDrowsiness() {
        fetch("/drowsiness_status")
            .then(response => {
                if (!response.ok) {
                    throw new Error("Network response was not ok");
                }
                return response.json();
            })
            .then(data => {
                statusElement.innerText = data.message;
            })
            .catch(error => {
                console.error("Error fetching drowsiness status:", error);
                statusElement.innerText = "Error fetching status";
            });
    }

    // Start polling every 1 second
    setInterval(checkDrowsiness, 1000);
});
