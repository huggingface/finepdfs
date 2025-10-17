HTML_DEFAULT_HEAD: str = r"""<head>
<link rel="icon" type="image/png"
href="https://ds4sd.github.io/docling/assets/logo.png"/>
<meta charset="UTF-8">
<title>
Powered by Docling
</title>
<style>
html {
background-color: LightGray;
}
body {
margin: 0 auto;
width:800px;
padding: 30px;
background-color: White;
font-family: Arial, sans-serif;
box-shadow: 10px 10px 10px grey;
}
figure{
display: block;
width: 100%;
margin: 0px;
margin-top: 10px;
margin-bottom: 10px;
}
img {
display: block;
margin: auto;
margin-top: 10px;
margin-bottom: 10px;
max-width: 640px;
max-height: 640px;
}
table {
min-width:500px;
background-color: White;
border-collapse: collapse;
cell-padding: 5px;
margin: auto;
margin-top: 10px;
margin-bottom: 10px;
}
th, td {
border: 1px solid black;
padding: 8px;
}
th {
font-weight: bold;
}
table tr:nth-child(even) td{
background-color: LightGray;
}
</style>
</head>"""

HTML_DEFAULT_HEAD_FOR_COMP: str = r"""<head>
<link rel="icon" type="image/png"
href="https://ds4sd.github.io/docling/assets/logo.png"/>
<meta charset="UTF-8">
<title>
Powered by Docling
</title>
<style>
html {
background-color: LightGray;
}
body {
margin: 0 auto;
padding: 10px;
background-color: White;
font-family: Arial, sans-serif;
box-shadow: 10px 10px 10px grey;
}
figure{
display: block;
width: 100%;
margin: 0px;
margin-top: 10px;
margin-bottom: 10px;
}
img {
display: block;
margin: auto;
margin-top: 10px;
margin-bottom: 10px;
max-width: 640px;
max-height: 640px;
}
table {
min-width:500px;
background-color: White;
border-collapse: collapse;
cell-padding: 5px;
margin: auto;
margin-top: 10px;
margin-bottom: 10px;
}
th, td {
border: 1px solid black;
padding: 8px;
}
th {
font-weight: bold;
}
table tr:nth-child(even) td{
background-color: LightGray;
}
</style>
</head>"""


HTML_COMPARISON_PAGE_v1 = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Side by Side Layout</title>
    <style>
        body {
            display: flex;
            justify-content: space-around; /* Adjust spacing between items */
            align-items: flex-start; /* Align items at the top */
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }
        iframe, img {
            width: 30%; /* Adjust the width of each item */
            height: 1024; /* Set a fixed height */
            border: 1px solid #ccc; /* Optional: Add borders */
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* Optional: Add shadow */
        }
        iframe {
            overflow: auto; /* Add scrolling if the content is larger */
        }
    </style>
</head>
<body>
    <!-- Image -->
    <img src="data:image/png;base64,BASE64PAGE" alt="Page Image">
    <!-- First HTML page -->
    <iframe srcdoc='TRUEDOC' title="Ground Truth"></iframe>
    <!-- Second HTML page -->
    <iframe srcdoc='PREDDOC' title="Docling Result"></iframe>
</body>
</html>
"""


HTML_COMPARISON_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Contained Page with Titles</title>
    <style>
        body {
            display: flex;
            justify-content: space-around; /* Adjust spacing between items */
            align-items: flex-start; /* Align items at the top */
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            height: 100vh; /* Full viewport height */
            overflow: hidden; /* Prevent scrollbars from appearing unnecessarily */
        }
        .container {
            display: flex;
            flex-direction: column;
            width: 30%; /* Adjust the width of each item */
            height: 100%; /* Adjust height to fill parent container */
            border: 1px solid #ccc; /* Optional: Add borders */
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* Optional: Add shadow */
            background-color: #fff; /* Optional: Add background */
            overflow: hidden; /* Prevent content overflow */
        }
        .title {
            text-align: center;
            font-weight: bold;
            padding: 10px;
            background-color: #eee;
            border-bottom: 1px solid #ccc;
        }
        iframe {
            flex-grow: 1; /* Fill the remaining space */
            border: none; /* Remove iframe border */
        }
        img {
            width: 100%;
            height: auto; /* Keep the aspect ratio of the image */
        }
    </style>
</head>
<body>
    <!-- Embedded Image -->
    <div class="container">
        <div class="title">Page Image</div>
        <img src="data:image/png;base64,BASE64PAGE" alt="Example Image">
    </div>

    <!-- First HTML page -->
    <div class="container">
        <div class="title">GroundTruth</div>
        <iframe srcdoc='TRUEDOC' title="Page 1"></iframe>
    </div>

    <!-- Second HTML page -->
    <div class="container">
        <div class="title">Prediction</div>
        <iframe srcdoc='PREDDOC' title="Page 2"></iframe>
    </div>
</body>
</html>
"""

HTML_INSPECTION = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Contained Page with Titles</title>
    <style>
        body {
            display: flex;
            justify-content: space-around; /* Adjust spacing between items */
            align-items: flex-start; /* Align items at the top */
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            height: 100vh; /* Full viewport height */
            overflow: hidden; /* Prevent scrollbars from appearing unnecessarily */
        }
        .container {
            display: flex;
            flex-direction: column;
            width: 50%; /* Adjust the width of each item */
            height: 100%; /* Adjust height to fill parent container */
            border: 1px solid #ccc; /* Optional: Add borders */
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* Optional: Add shadow */
            background-color: #fff; /* Optional: Add background */
            overflow: auto; /* Prevent content overflow */
        }
        .title {
            text-align: center;
            font-weight: bold;
            padding: 10px;
            background-color: #eee;
            border-bottom: 1px solid #ccc;
        }
        .image-wrapper {
            padding: 10px; /* Add some space around each image */
            text-align: center;
        }
        .image-wrapper img {
            width: 100%; /* Scale images to container width */
            height: auto; /* Maintain aspect ratio */
            border: 1px solid #ccc;
            box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
        }
        iframe {
            flex-grow: 1; /* Fill the remaining space */
            border: none; /* Remove iframe border */
        }
        img {
            width: 100%;
            height: auto; /* Keep the aspect ratio of the image */
        }
    </style>
</head>
<body>
    <!-- Embedded Image -->
    <div class="container">
        <div class="title">Page Images</div>
        PAGE_IMAGES
    </div>

    <!-- Second HTML page -->
    <div class="container">
        <div class="title">Prediction</div>
        <iframe srcdoc='PREDDOC' title="Page 2"></iframe>
    </div>
</body>
</html>
"""


HTML_COMPARISON_PAGE_WITH_CLUSTERS = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Self-Contained Page with Titles</title>
    <style>
        body {
            display: flex;
            justify-content: space-around; /* Adjust spacing between items */
            align-items: flex-start; /* Align items at the top */
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            height: 100vh; /* Full viewport height */
            overflow: hidden; /* Prevent scrollbars from appearing unnecessarily */
        }
        .container {
            display: flex;
            flex-direction: column;
            width: 25%; /* Adjust the width of each item */
            height: 100%; /* Adjust height to fill parent container */
            border: 1px solid #ccc; /* Optional: Add borders */
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* Optional: Add shadow */
            background-color: #fff; /* Optional: Add background */
            overflow: hidden; /* Prevent content overflow */
        }
        .title {
            text-align: center;
            font-weight: bold;
            padding: 10px;
            background-color: #eee;
            border-bottom: 1px solid #ccc;
        }
        iframe {
            flex-grow: 1; /* Fill the remaining space */
            border: none; /* Remove iframe border */
        }
        img {
            width: 100%;
            height: auto; /* Keep the aspect ratio of the image */
        }
    </style>
</head>
<body>
    <!-- Embedded Image -->
    <div class="container">
        <div class="title">Page Image Groundtruth</div>
        <img src="data:image/png;base64,BASE64TRUEPAGE" alt="Example Image">
    </div>

    <!-- First HTML page -->
    <div class="container">
        <div class="title">GroundTruth</div>
        <iframe srcdoc='TRUEDOC' title="Page 1"></iframe>
    </div>

    <!-- Embedded Image -->
    <div class="container">
        <div class="title">Page Image Prediction</div>
        <img src="data:image/png;base64,BASE64PREDPAGE" alt="Example Image">
    </div>

    <!-- Second HTML page -->
    <div class="container">
        <div class="title">Prediction</div>
        <iframe srcdoc='PREDDOC' title="Page 2"></iframe>
    </div>
</body>
</html>
"""
