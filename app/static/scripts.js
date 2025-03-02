document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file");

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            alert("File selected: " + fileInput.files[0].name);
        }
    });
});
