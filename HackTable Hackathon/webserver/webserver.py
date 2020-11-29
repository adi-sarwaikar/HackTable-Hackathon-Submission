from flask import Flask, render_template, request
from api_user import main_function

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        county = request.form["county"]
        half_year, half_year_response, full_year, full_year_response = main_function(county)
        return "<html><head><style>html {font-size: 100%;-webkit-text-size-adjust: 100%;-ms-text-size-adjust: 100%;}html, button, form, select, textarea {font-family: sans-serif;color: #222;}body {font-family: Arial, Helvetica, sans-serif;padding: 0px;margin: 0px;background-color: rgba(230, 230, 230, 0.85) !important;}#wrapper-sc * {-webkit-box-sizing: border-box;/* Needed for mobile WebKit */-moz-box-sizing: border-box;/* Needed for Firefox */box-sizing: border-box;}#wrapper-sc .sc-validation-state\:invalid {color: black;background: #F05050;}#wrapper-sc .sc-validation-state\:valid {outline: 2px solid #A9E2F3;color: black;}#wrapper-sc .sc-validation-state\:default {outline: 2px solid transparent;color: black;}#wrapper-sc .sc-validation-state\:invalid::-webkit-input-placeholder {color: black;}#wrapper-sc .sc-validation-state\:invalid:-moz-placeholder { /* Firefox 18- */color: black;}#wrapper-sc .sc-validation-state\:invalid::-moz-placeholder {  /* Firefox 19+ */color: black;}#wrapper-sc .sc-validation-state\:invalid:-ms-form-placeholder {color: black;}#wrapper-sc .emailvalidation {color: #E8C937;text-align: center;float: left;padding: 0px 20px 8px 20px;width: 100%;font-weight: bold;}#wrapper-sc .group-sc:before, .group-sc:after {display: table;content: "";}#wrapper-sc .group-sc:after {clear: both;}#wrapper-sc::-moz-selection {background: #000000;color: #fff;text-shadow: none;}#wrapper-sc::selection {background: #000000;color: #fff;text-shadow: none;}#wrapper-sc {animation: animationFrames ease 1s;animation-iteration-count: 1;transform-origin: 50% 50%;animation-fill-mode: forwards;/*when the spec is finished*/-webkit-animation: animationFrames ease 1s;-webkit-animation-iteration-count: 1;-webkit-transform-origin: 50% 50%;-webkit-animation-fill-mode: forwards;/*Chrome 16+, Safari 4+*/-moz-animation: animationFrames ease 1s;-moz-animation-iteration-count: 1;-moz-transform-origin: 50% 50%;-moz-animation-fill-mode: forwards;/*FF 5+*/-o-animation: animationFrames ease 1s;-o-animation-iteration-count: 1;-o-transform-origin: 50% 50%;-o-animation-fill-mode: forwards;/*Not implemented yet*/-ms-animation: animationFrames ease 1s;-ms-animation-iteration-count: 1;-ms-transform-origin: 50% 50%;-ms-animation-fill-mode: forwards;/*IE 10+*/}@keyframes animationFrames {0% {opacity: 0;transform: translate(0px, -25px);}100% {opacity: 1;transform: translate(0px, 0px);}}@-moz-keyframes animationFrames {0% {opacity: 0;-moz-transform: translate(0px, -25px);}100% {opacity: 1;-moz-transform: translate(0px, 0px);}}@-webkit-keyframes animationFrames {0% {opacity: 0;-webkit-transform: translate(0px, -25px);}100% {opacity: 1;-webkit-transform: translate(0px, 0px);}}@-o-keyframes animationFrames {0% {opacity: 0;-o-transform: translate(0px, -25px);}100% {opacity: 1;-o-transform: translate(0px, 0px);}}@-ms-keyframes animationFrames {0% {opacity: 0;-ms-transform: translate(0px, -25px);}100% {opacity: 1;-ms-transform: translate(0px, 0px);}}#wrapper-sc p, #wrapper-sc a, #wrapper-sc h1, #wrapper-sc h2, #wrapper-sc h3, #wrapper-sc h4, #wrapper-sc ul, #wrapper-sc ol, #wrapper-sc dd, #wrapper-sc nav ul, #wrapper-sc nav ol {font-family: sans-serif;}#wrapper-sc p {margin: 0px;padding: 0px;font-size: 12px;line-height: 15px;}#wrapper-sc a {margin: 0px;padding: 0px;font-size: 12px;line-height: 15px;text-decoration: none;color: #FFFFFF;}#wrapper-sc h1 {font-family: 'Avenir_Reg', arial, sans-serif;margin: 0px;padding: 0px;font-size: 32px;line-height: 35px;text-align: center;font-weight: lighter;color:#0d0e0e;}#wrapper-sc h2 {font-family: 'Avenir_Reg', arial, sans-serif;margin: 15px 0px 0px 0px;padding: 0px;font-size: 16px;line-height: 20px;text-align: center;font-weight: lighter;color: #333333;}#wrapper-sc h3 {font-family: 'Avenir_Reg', arial, sans-serif;margin: 0px;padding: 0px;font-size: 12px;line-height: 16px;text-align: center;font-weight: lighter;color: #afafaf;}#wrapper-sc h4 {margin: 0px;padding: 0px;font-size: 14px;line-height: 18px;}#wrapper-sc h6 {margin: 0px;padding: 0px;font-size: 10px;line-height: 12px;color: #FFFFFF;font-weight: lighter;text-align: justify;}#wrapper-sc ul, #wrapper-sc ol {margin: 0 0;padding: 0 0 0 0;}#wrapper-sc dd {	margin: 0 0 0 0;}#wrapper-sc nav ul, #wrapper-sc nav ol {margin: 0;padding: 0;}/*This stops the initial white background issue and set the padding above the OSR*/.osr-content {background-color: transparent !important;top: 15% !important;width: 320px;/* this needs to match the wrapper-sc width */}.osr-overbackground-color: rgba(230, 230, 230, 0#wrapper-sc input:focus::-webkit-input-placeholder {color: transparent;border-radius: 400px 400px 400px 400px;color: #d2d2d2;}#wrapper-sc input:focus:-moz-placeholder {color: transparent;-moz-border-radius: 400px 400px 400px 400px;color: #d2d2d2;}/* FF 4-#wrapper-sc input:focus::-moz-placeholcolor: transpa-moz-border-radius: 400px 400px 400px 4color: #d2}/* FF 1#wrapper-sc input:focus:-ms-input-placeholcolor: transpa-ms-border-radius: 400px 400px 400px 4}/* IE 1@font-ffont-family: 'Avenir_font-style: nofont-weight: ligsrc: url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextRegular.esrc: url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextRegular.eot') format('embedded-opentype'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextRegular.woff2') format('woff2'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextRegular.woff') format('woff'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextRegular.ttf') format('truetype'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextRegular.svg') format('s@font-ffont-family: 'Avenir_Bfont-style: nofont-weight: nosrc: url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextLTProDemi.esrc: url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextLTProDemi.eot') format('embedded-opentype'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextLTProDemi.woff2') format('woff2'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextLTProDemi.woff') format('woff'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextLTProDemi.ttf') format('truetype'),  url('https://cdn-salecycle.s3.amazonaws.com/images/fontlibrary/AvenirNextLTProDemi.svg') format('s#wrapperposition: relafont-family: sans-sdisplay: bwidth: 4box-sizing: borderposition: relamargin: 10px auto 0px au#wrapper-sc .contentpadding: 100px 40px 0px #wrapper-sc .contentbackpadding: 120px 40px 0px #wrapper-sc .nothanksmargin-top:padding: 0px 20px 20px position:absobottom:display:b#wrapper-sc .front-sc, #wrapper-sc .backwidth: 4height: 3padding:margin-top: border-radius: 10px 10px 10px -moz-border-radius: 10px 10px 10px -webkit-border-radius: 10px 10px 10px /* Needs to match wrapper-sc wi#wrapper-sc .front-sc, #wrapper-sc .back-webkit-backface-visibility: hi-moz-backface-visibility: hi-ms-backface-visibility: hibackface-visibility: hi-webkit-transition:-webkit-transform-style: preserv-moz-transition:-moz-transform-style: preserv-o-transition:-o-transform-style: preserv-ms-transition:-ms-transform-style: preservtransition:transform-style: preservposition: absotolef.front-f-webkit-transform: rotateY(0-moz-transform: rotateY(0-o-transform: rotateY(0-ms-transform: rotateY(0transform: rotateY(0#wrapper-sc .frontz-indewidth: 4height: 3background-color: #ffbackground-position:background-repeat: no-remargin-top: #wrapper-sc .iconheight: 1width: 1position: absotop: -left: -1righmargin-left: margin-right: #wrapper-sc inwidth: 3border: 2px solid #52height: background-color: padding: 0px 0px 0px margin: 0px font-size: font-family: Arial, sans-sfont-weight: ligcolor: #30display: bvertical-align: mioutline: none!imporbox-shadow: none!impor-webkit-border-radius: 400px 400px 400px 400px!impor-moz-border-radius: 400px 400px 400px 400px!imporborder-radius: 400px 400px 400px 400px!imporbox-sizing: border-box!importext-align: ce#wrapper-sc input:requibox-shadow:none!imporoutline: none!impor-webkit-border-radius: 400px 400px 400px 400px!impor-moz-border-radius: 400px 400px 400px 400px!imporborder-radius: 400px 400px 400px 400px!imporbox-sizing: border-box!impor#wrapper-sc input:invabox-shadow:none!imporoutline: none!impor-webkit-border-radius: 400px 400px 400px 400px!impor-moz-border-radius: 400px 400px 400px 400px!imporborder-radius: 400px 400px 400px 400px!imporbox-sizing: border-box!impor#wrapper-sc .emailboxdisplay: bpadding-top: padding-bottom: #wrapper-sc .btndisplay: bmargin: 10px auto 0px background-color: #d5text-align: cefont-size: color: #FFfont-family: 'Avenir_Bold', arial, sans-sfont-weight: font-style: height: line-height: width: 2transition: all .2s ease-in-webkit-border-radius: 400px 400px 400px 400px!impor-moz-border-radius: 400px 400px 400px 400px!imporborder-radius: 400px 400px 400px 400px!impor#wrapper-sc .btn-sc:hobackground-color: #B0}@media screen and (max-width: 480px) {/* Responsive Styling below this line - change the width above depending on OSR Width */.osr-content {	width: 100%!important;}#wrapper-sc {	width: 100%;}#wrapper-sc .exit-sc {	top: 10px !important;	right: 0px !important;}#wrapper-sc h1 {	font-family: 'Avenir_Reg', arial, sans-serif;	margin: 0px;	padding: 0px;	font-size: 22px;	line-height: 25px;	text-align: center;	font-weight: lighter;}#wrapper-sc h2 {	font-family: 'Avenir_Reg', arial, sans-serif;	margin: 0px !important;	padding: 0px !important;	font-size: 12px;	line-height: 16px;	text-align: center;	font-weight: lighter;	color: #afafaf;}#wrapper-sc .vanish {	display: none;}#wrapper-sc .front-sc {	width: 100%;	height: 320px;}#wrapper-sc .content-sc {	padding-top: 100px;}#wrapper-sc input {	width: 90%;	border: 1px solid #52caef;}#wrapper-sc .btn-sc {	width: 90% !important;}#wrapper-sc .nothanks-sc {	margin-top: 0px;	padding: 0px 20px 20px 20px;	position:absolute;	bottom:-5px;	display:block;}}</style></head><body><div style=\"display:none !important;\"> </div><div id=\"wrapper-sc\">       <div class=\"front-sc front-flip\"> <a class=\"exit-sc close-sc\" href=\"javascript:void(0);\">×</a>    <div class=\"icon-sc\"> </div><div class=\"content-sc\"><h1> <span class=\"vanish\"> </br> Expected number of coronavirus cases in "+county+" county: </br> </br> 1 month from now: "+ str(half_year) + " Outbreak occuring: " + str(half_year_response) + "</br> 3 months from now:"+ str(full_year) + " Outbreak occuring: " + str(full_year_response) + "</h1></div><div class=\"emailContainer-sc\" id=\"emailContainer-sc\"><div class=\"emailbox-sc\"><div class=\"emailboxelement-sc\"><!--<form> <input class=\"county-field-sc emb-sc\" type=\"county\" placeholder=\"enter your county here\" > </form> </div><a class=\"send-sc btn-sc\" href=\"javascript:void(0);\">Submit</a> </div> --></div><div class=\"nothanks-sc\"><h3> <em></em></h3></div></div></div></body></html>"
    else:
        return render_template('main.html')

if __name__ == '__main__':
    app.run()