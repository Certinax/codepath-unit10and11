# codepath-unit10and11

<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />

<title>income</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>



<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.7.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.7.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.7.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff2?v=4.7.0') format('woff2'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.7.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.7.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.7.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.fa-pull-left {
  float: left;
}
.fa-pull-right {
  float: right;
}
.fa.fa-pull-left {
  margin-right: .3em;
}
.fa.fa-pull-right {
  margin-left: .3em;
}
/* Deprecated as of 4.4.0 */
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
.fa-pulse {
  -webkit-animation: fa-spin 1s infinite steps(8);
  animation: fa-spin 1s infinite steps(8);
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook-f:before,
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-feed:before,
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before,
.fa-gratipay:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper-pp:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-resistance:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-y-combinator-square:before,
.fa-yc-square:before,
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
.fa-buysellads:before {
  content: "\f20d";
}
.fa-connectdevelop:before {
  content: "\f20e";
}
.fa-dashcube:before {
  content: "\f210";
}
.fa-forumbee:before {
  content: "\f211";
}
.fa-leanpub:before {
  content: "\f212";
}
.fa-sellsy:before {
  content: "\f213";
}
.fa-shirtsinbulk:before {
  content: "\f214";
}
.fa-simplybuilt:before {
  content: "\f215";
}
.fa-skyatlas:before {
  content: "\f216";
}
.fa-cart-plus:before {
  content: "\f217";
}
.fa-cart-arrow-down:before {
  content: "\f218";
}
.fa-diamond:before {
  content: "\f219";
}
.fa-ship:before {
  content: "\f21a";
}
.fa-user-secret:before {
  content: "\f21b";
}
.fa-motorcycle:before {
  content: "\f21c";
}
.fa-street-view:before {
  content: "\f21d";
}
.fa-heartbeat:before {
  content: "\f21e";
}
.fa-venus:before {
  content: "\f221";
}
.fa-mars:before {
  content: "\f222";
}
.fa-mercury:before {
  content: "\f223";
}
.fa-intersex:before,
.fa-transgender:before {
  content: "\f224";
}
.fa-transgender-alt:before {
  content: "\f225";
}
.fa-venus-double:before {
  content: "\f226";
}
.fa-mars-double:before {
  content: "\f227";
}
.fa-venus-mars:before {
  content: "\f228";
}
.fa-mars-stroke:before {
  content: "\f229";
}
.fa-mars-stroke-v:before {
  content: "\f22a";
}
.fa-mars-stroke-h:before {
  content: "\f22b";
}
.fa-neuter:before {
  content: "\f22c";
}
.fa-genderless:before {
  content: "\f22d";
}
.fa-facebook-official:before {
  content: "\f230";
}
.fa-pinterest-p:before {
  content: "\f231";
}
.fa-whatsapp:before {
  content: "\f232";
}
.fa-server:before {
  content: "\f233";
}
.fa-user-plus:before {
  content: "\f234";
}
.fa-user-times:before {
  content: "\f235";
}
.fa-hotel:before,
.fa-bed:before {
  content: "\f236";
}
.fa-viacoin:before {
  content: "\f237";
}
.fa-train:before {
  content: "\f238";
}
.fa-subway:before {
  content: "\f239";
}
.fa-medium:before {
  content: "\f23a";
}
.fa-yc:before,
.fa-y-combinator:before {
  content: "\f23b";
}
.fa-optin-monster:before {
  content: "\f23c";
}
.fa-opencart:before {
  content: "\f23d";
}
.fa-expeditedssl:before {
  content: "\f23e";
}
.fa-battery-4:before,
.fa-battery:before,
.fa-battery-full:before {
  content: "\f240";
}
.fa-battery-3:before,
.fa-battery-three-quarters:before {
  content: "\f241";
}
.fa-battery-2:before,
.fa-battery-half:before {
  content: "\f242";
}
.fa-battery-1:before,
.fa-battery-quarter:before {
  content: "\f243";
}
.fa-battery-0:before,
.fa-battery-empty:before {
  content: "\f244";
}
.fa-mouse-pointer:before {
  content: "\f245";
}
.fa-i-cursor:before {
  content: "\f246";
}
.fa-object-group:before {
  content: "\f247";
}
.fa-object-ungroup:before {
  content: "\f248";
}
.fa-sticky-note:before {
  content: "\f249";
}
.fa-sticky-note-o:before {
  content: "\f24a";
}
.fa-cc-jcb:before {
  content: "\f24b";
}
.fa-cc-diners-club:before {
  content: "\f24c";
}
.fa-clone:before {
  content: "\f24d";
}
.fa-balance-scale:before {
  content: "\f24e";
}
.fa-hourglass-o:before {
  content: "\f250";
}
.fa-hourglass-1:before,
.fa-hourglass-start:before {
  content: "\f251";
}
.fa-hourglass-2:before,
.fa-hourglass-half:before {
  content: "\f252";
}
.fa-hourglass-3:before,
.fa-hourglass-end:before {
  content: "\f253";
}
.fa-hourglass:before {
  content: "\f254";
}
.fa-hand-grab-o:before,
.fa-hand-rock-o:before {
  content: "\f255";
}
.fa-hand-stop-o:before,
.fa-hand-paper-o:before {
  content: "\f256";
}
.fa-hand-scissors-o:before {
  content: "\f257";
}
.fa-hand-lizard-o:before {
  content: "\f258";
}
.fa-hand-spock-o:before {
  content: "\f259";
}
.fa-hand-pointer-o:before {
  content: "\f25a";
}
.fa-hand-peace-o:before {
  content: "\f25b";
}
.fa-trademark:before {
  content: "\f25c";
}
.fa-registered:before {
  content: "\f25d";
}
.fa-creative-commons:before {
  content: "\f25e";
}
.fa-gg:before {
  content: "\f260";
}
.fa-gg-circle:before {
  content: "\f261";
}
.fa-tripadvisor:before {
  content: "\f262";
}
.fa-odnoklassniki:before {
  content: "\f263";
}
.fa-odnoklassniki-square:before {
  content: "\f264";
}
.fa-get-pocket:before {
  content: "\f265";
}
.fa-wikipedia-w:before {
  content: "\f266";
}
.fa-safari:before {
  content: "\f267";
}
.fa-chrome:before {
  content: "\f268";
}
.fa-firefox:before {
  content: "\f269";
}
.fa-opera:before {
  content: "\f26a";
}
.fa-internet-explorer:before {
  content: "\f26b";
}
.fa-tv:before,
.fa-television:before {
  content: "\f26c";
}
.fa-contao:before {
  content: "\f26d";
}
.fa-500px:before {
  content: "\f26e";
}
.fa-amazon:before {
  content: "\f270";
}
.fa-calendar-plus-o:before {
  content: "\f271";
}
.fa-calendar-minus-o:before {
  content: "\f272";
}
.fa-calendar-times-o:before {
  content: "\f273";
}
.fa-calendar-check-o:before {
  content: "\f274";
}
.fa-industry:before {
  content: "\f275";
}
.fa-map-pin:before {
  content: "\f276";
}
.fa-map-signs:before {
  content: "\f277";
}
.fa-map-o:before {
  content: "\f278";
}
.fa-map:before {
  content: "\f279";
}
.fa-commenting:before {
  content: "\f27a";
}
.fa-commenting-o:before {
  content: "\f27b";
}
.fa-houzz:before {
  content: "\f27c";
}
.fa-vimeo:before {
  content: "\f27d";
}
.fa-black-tie:before {
  content: "\f27e";
}
.fa-fonticons:before {
  content: "\f280";
}
.fa-reddit-alien:before {
  content: "\f281";
}
.fa-edge:before {
  content: "\f282";
}
.fa-credit-card-alt:before {
  content: "\f283";
}
.fa-codiepie:before {
  content: "\f284";
}
.fa-modx:before {
  content: "\f285";
}
.fa-fort-awesome:before {
  content: "\f286";
}
.fa-usb:before {
  content: "\f287";
}
.fa-product-hunt:before {
  content: "\f288";
}
.fa-mixcloud:before {
  content: "\f289";
}
.fa-scribd:before {
  content: "\f28a";
}
.fa-pause-circle:before {
  content: "\f28b";
}
.fa-pause-circle-o:before {
  content: "\f28c";
}
.fa-stop-circle:before {
  content: "\f28d";
}
.fa-stop-circle-o:before {
  content: "\f28e";
}
.fa-shopping-bag:before {
  content: "\f290";
}
.fa-shopping-basket:before {
  content: "\f291";
}
.fa-hashtag:before {
  content: "\f292";
}
.fa-bluetooth:before {
  content: "\f293";
}
.fa-bluetooth-b:before {
  content: "\f294";
}
.fa-percent:before {
  content: "\f295";
}
.fa-gitlab:before {
  content: "\f296";
}
.fa-wpbeginner:before {
  content: "\f297";
}
.fa-wpforms:before {
  content: "\f298";
}
.fa-envira:before {
  content: "\f299";
}
.fa-universal-access:before {
  content: "\f29a";
}
.fa-wheelchair-alt:before {
  content: "\f29b";
}
.fa-question-circle-o:before {
  content: "\f29c";
}
.fa-blind:before {
  content: "\f29d";
}
.fa-audio-description:before {
  content: "\f29e";
}
.fa-volume-control-phone:before {
  content: "\f2a0";
}
.fa-braille:before {
  content: "\f2a1";
}
.fa-assistive-listening-systems:before {
  content: "\f2a2";
}
.fa-asl-interpreting:before,
.fa-american-sign-language-interpreting:before {
  content: "\f2a3";
}
.fa-deafness:before,
.fa-hard-of-hearing:before,
.fa-deaf:before {
  content: "\f2a4";
}
.fa-glide:before {
  content: "\f2a5";
}
.fa-glide-g:before {
  content: "\f2a6";
}
.fa-signing:before,
.fa-sign-language:before {
  content: "\f2a7";
}
.fa-low-vision:before {
  content: "\f2a8";
}
.fa-viadeo:before {
  content: "\f2a9";
}
.fa-viadeo-square:before {
  content: "\f2aa";
}
.fa-snapchat:before {
  content: "\f2ab";
}
.fa-snapchat-ghost:before {
  content: "\f2ac";
}
.fa-snapchat-square:before {
  content: "\f2ad";
}
.fa-pied-piper:before {
  content: "\f2ae";
}
.fa-first-order:before {
  content: "\f2b0";
}
.fa-yoast:before {
  content: "\f2b1";
}
.fa-themeisle:before {
  content: "\f2b2";
}
.fa-google-plus-circle:before,
.fa-google-plus-official:before {
  content: "\f2b3";
}
.fa-fa:before,
.fa-font-awesome:before {
  content: "\f2b4";
}
.fa-handshake-o:before {
  content: "\f2b5";
}
.fa-envelope-open:before {
  content: "\f2b6";
}
.fa-envelope-open-o:before {
  content: "\f2b7";
}
.fa-linode:before {
  content: "\f2b8";
}
.fa-address-book:before {
  content: "\f2b9";
}
.fa-address-book-o:before {
  content: "\f2ba";
}
.fa-vcard:before,
.fa-address-card:before {
  content: "\f2bb";
}
.fa-vcard-o:before,
.fa-address-card-o:before {
  content: "\f2bc";
}
.fa-user-circle:before {
  content: "\f2bd";
}
.fa-user-circle-o:before {
  content: "\f2be";
}
.fa-user-o:before {
  content: "\f2c0";
}
.fa-id-badge:before {
  content: "\f2c1";
}
.fa-drivers-license:before,
.fa-id-card:before {
  content: "\f2c2";
}
.fa-drivers-license-o:before,
.fa-id-card-o:before {
  content: "\f2c3";
}
.fa-quora:before {
  content: "\f2c4";
}
.fa-free-code-camp:before {
  content: "\f2c5";
}
.fa-telegram:before {
  content: "\f2c6";
}
.fa-thermometer-4:before,
.fa-thermometer:before,
.fa-thermometer-full:before {
  content: "\f2c7";
}
.fa-thermometer-3:before,
.fa-thermometer-three-quarters:before {
  content: "\f2c8";
}
.fa-thermometer-2:before,
.fa-thermometer-half:before {
  content: "\f2c9";
}
.fa-thermometer-1:before,
.fa-thermometer-quarter:before {
  content: "\f2ca";
}
.fa-thermometer-0:before,
.fa-thermometer-empty:before {
  content: "\f2cb";
}
.fa-shower:before {
  content: "\f2cc";
}
.fa-bathtub:before,
.fa-s15:before,
.fa-bath:before {
  content: "\f2cd";
}
.fa-podcast:before {
  content: "\f2ce";
}
.fa-window-maximize:before {
  content: "\f2d0";
}
.fa-window-minimize:before {
  content: "\f2d1";
}
.fa-window-restore:before {
  content: "\f2d2";
}
.fa-times-rectangle:before,
.fa-window-close:before {
  content: "\f2d3";
}
.fa-times-rectangle-o:before,
.fa-window-close-o:before {
  content: "\f2d4";
}
.fa-bandcamp:before {
  content: "\f2d5";
}
.fa-grav:before {
  content: "\f2d6";
}
.fa-etsy:before {
  content: "\f2d7";
}
.fa-imdb:before {
  content: "\f2d8";
}
.fa-ravelry:before {
  content: "\f2d9";
}
.fa-eercast:before {
  content: "\f2da";
}
.fa-microchip:before {
  content: "\f2db";
}
.fa-snowflake-o:before {
  content: "\f2dc";
}
.fa-superpowers:before {
  content: "\f2dd";
}
.fa-wpexplorer:before {
  content: "\f2de";
}
.fa-meetup:before {
  content: "\f2e0";
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
div.traceback-wrapper pre.traceback {
  max-height: 600px;
  overflow: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  padding: 5px;
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
[dir="rtl"] #ipython_notebook {
  margin-right: 10px;
  margin-left: 0;
}
[dir="rtl"] #ipython_notebook.pull-left {
  float: right !important;
  float: right;
}
.flex-spacer {
  flex: 1;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#kernel_logo_widget {
  margin: 0 10px;
}
span#login_widget {
  float: right;
}
[dir="rtl"] span#login_widget {
  float: left;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
.modal-header {
  cursor: move;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
[dir="rtl"] .center-nav form.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] .center-nav .navbar-text {
  float: right;
}
[dir="rtl"] .navbar-inner {
  text-align: right;
}
[dir="rtl"] div.text-left {
  text-align: right;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  position: absolute;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: pointer;
  opacity: 0;
  z-index: 2;
}
.alternate_upload .btn-xs > input.fileinput {
  margin: -1px -5px;
}
.alternate_upload .btn-upload {
  position: relative;
  height: 22px;
}
::-webkit-file-upload-button {
  cursor: pointer;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
[dir="rtl"] ul#tabs.nav-tabs > li {
  float: right;
}
[dir="rtl"] ul#tabs.nav.nav-tabs {
  padding-right: 0;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons .pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .list_toolbar .col-sm-4,
[dir="rtl"] .list_toolbar .col-sm-8 {
  float: right;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: text-bottom;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
[dir="rtl"] .list_item > div input {
  margin-right: 0;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_modified {
  margin-right: 7px;
  margin-left: 7px;
}
[dir="rtl"] .item_modified.pull-right {
  float: left !important;
  float: left;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
[dir="rtl"] .item_buttons.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .item_buttons .kernel-name {
  margin-left: 7px;
  float: right;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
.sort_button {
  display: inline-block;
  padding-left: 7px;
}
[dir="rtl"] .sort_button.pull-right {
  float: left !important;
  float: left;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
[dir="rtl"] #button-select-all.btn {
  float: right ;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
  margin-top: 2px;
  height: 16px;
}
[dir="rtl"] #select-all.pull-left {
  float: right !important;
  float: right;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.fa-pull-left {
  margin-right: .3em;
}
.folder_icon:before.fa-pull-right {
  margin-left: .3em;
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.fa-pull-left {
  margin-right: .3em;
}
.file_icon:before.fa-pull-right {
  margin-left: .3em;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
#new-menu .dropdown-header {
  font-size: 10px;
  border-bottom: 1px solid #e5e5e5;
  padding: 0 0 3px;
  margin: -3px 20px 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.move-button {
  display: none;
}
.download-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
.CodeMirror-dialog {
  background-color: #fff;
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI escape sequences */
/* The color values are a mix of
   http://www.xcolors.net/dl/baskerville-ivorylight and
   http://www.xcolors.net/dl/euphrasia */
.ansi-black-fg {
  color: #3E424D;
}
.ansi-black-bg {
  background-color: #3E424D;
}
.ansi-black-intense-fg {
  color: #282C36;
}
.ansi-black-intense-bg {
  background-color: #282C36;
}
.ansi-red-fg {
  color: #E75C58;
}
.ansi-red-bg {
  background-color: #E75C58;
}
.ansi-red-intense-fg {
  color: #B22B31;
}
.ansi-red-intense-bg {
  background-color: #B22B31;
}
.ansi-green-fg {
  color: #00A250;
}
.ansi-green-bg {
  background-color: #00A250;
}
.ansi-green-intense-fg {
  color: #007427;
}
.ansi-green-intense-bg {
  background-color: #007427;
}
.ansi-yellow-fg {
  color: #DDB62B;
}
.ansi-yellow-bg {
  background-color: #DDB62B;
}
.ansi-yellow-intense-fg {
  color: #B27D12;
}
.ansi-yellow-intense-bg {
  background-color: #B27D12;
}
.ansi-blue-fg {
  color: #208FFB;
}
.ansi-blue-bg {
  background-color: #208FFB;
}
.ansi-blue-intense-fg {
  color: #0065CA;
}
.ansi-blue-intense-bg {
  background-color: #0065CA;
}
.ansi-magenta-fg {
  color: #D160C4;
}
.ansi-magenta-bg {
  background-color: #D160C4;
}
.ansi-magenta-intense-fg {
  color: #A03196;
}
.ansi-magenta-intense-bg {
  background-color: #A03196;
}
.ansi-cyan-fg {
  color: #60C6C8;
}
.ansi-cyan-bg {
  background-color: #60C6C8;
}
.ansi-cyan-intense-fg {
  color: #258F8F;
}
.ansi-cyan-intense-bg {
  background-color: #258F8F;
}
.ansi-white-fg {
  color: #C5C1B4;
}
.ansi-white-bg {
  background-color: #C5C1B4;
}
.ansi-white-intense-fg {
  color: #A1A6B2;
}
.ansi-white-intense-bg {
  background-color: #A1A6B2;
}
.ansi-default-inverse-fg {
  color: #FFFFFF;
}
.ansi-default-inverse-bg {
  background-color: #000000;
}
.ansi-bold {
  font-weight: bold;
}
.ansi-underline {
  text-decoration: underline;
}
/* The following styles are deprecated an will be removed in a future version */
.ansibold {
  font-weight: bold;
}
.ansi-inverse {
  outline: 0.5px dotted;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  position: relative;
  overflow: visible;
}
div.cell:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: transparent;
}
div.cell.jupyter-soft-selected {
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected,
div.cell.selected.jupyter-soft-selected {
  border-color: #ababab;
}
div.cell.selected:before,
div.cell.selected.jupyter-soft-selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #42A5F5;
}
@media print {
  div.cell.selected,
  div.cell.selected.jupyter-soft-selected {
    border-color: transparent;
  }
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
}
.edit_mode div.cell.selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #66BB6A;
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  /* Note that this should set vertical padding only, since CodeMirror assumes
       that horizontal padding will be set on CodeMirror pre */
  padding: 0.4em 0;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. This sets horizontal padding only,
    use .CodeMirror-lines for vertical */
  padding: 0 0.4em;
  border: 0;
  border-radius: 0;
}
.CodeMirror-cursor {
  border-left: 1.4px solid black;
}
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .CodeMirror-cursor {
    border-left: 2px solid black;
  }
}
@media screen and (min-width: 4320px) {
  .CodeMirror-cursor {
    border-left: 4px solid black;
  }
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
div.output_area .mglyph > img {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 1px 0 1px 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul:not(.list-inline),
.rendered_html ol:not(.list-inline) {
  padding-left: 2em;
}
.rendered_html ul {
  list-style: disc;
}
.rendered_html ul ul {
  list-style: square;
  margin-top: 0;
}
.rendered_html ul ul ul {
  list-style: circle;
}
.rendered_html ol {
  list-style: decimal;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin-top: 0;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
  padding: 0px;
  background-color: #fff;
}
.rendered_html code {
  background-color: #eff0f1;
}
.rendered_html p code {
  padding: 1px 5px;
}
.rendered_html pre code {
  background-color: #fff;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  color: #000;
  font-size: 100%;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: none;
  border-collapse: collapse;
  border-spacing: 0;
  color: black;
  font-size: 12px;
  table-layout: fixed;
}
.rendered_html thead {
  border-bottom: 1px solid black;
  vertical-align: bottom;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  text-align: right;
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html tbody tr:nth-child(odd) {
  background: #f5f5f5;
}
.rendered_html tbody tr:hover {
  background: rgba(66, 165, 245, 0.2);
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
.rendered_html .alert {
  margin-bottom: initial;
}
.rendered_html * + .alert {
  margin-top: 1em;
}
[dir="rtl"] .rendered_html p {
  text-align: right;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.rendered .rendered_html tr,
.text_cell.rendered .rendered_html th,
.text_cell.rendered .rendered_html td {
  max-width: none;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.text_cell .dropzone .input_area {
  border: 2px dashed #bababa;
  margin: -1px;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
.jupyter-keybindings {
  padding: 1px;
  line-height: 24px;
  border-bottom: 1px solid gray;
}
.jupyter-keybindings input {
  margin: 0;
  padding: 0;
  border: none;
}
.jupyter-keybindings i {
  padding: 6px;
}
.well code {
  background-color: #ffffff;
  border-color: #ababab;
  border-width: 1px;
  border-style: solid;
  padding: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.tags_button_container {
  width: 100%;
  display: flex;
}
.tag-container {
  display: flex;
  flex-direction: row;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
}
.tag-container > * {
  margin: 0 4px;
}
.remove-tag-btn {
  margin-left: 4px;
}
.tags-input {
  display: flex;
}
.cell-tag:last-child:after {
  content: "";
  position: absolute;
  right: 0;
  width: 40px;
  height: 100%;
  /* Fade to background color of cell toolbar */
  background: linear-gradient(to right, rgba(0, 0, 0, 0), #EEE);
}
.tags-input > * {
  margin-left: 4px;
}
.cell-tag,
.tags-input input,
.tags-input button {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  box-shadow: none;
  width: inherit;
  font-size: inherit;
  height: 22px;
  line-height: 22px;
  padding: 0px 4px;
  display: inline-block;
}
.cell-tag:focus,
.tags-input input:focus,
.tags-input button:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.cell-tag::-moz-placeholder,
.tags-input input::-moz-placeholder,
.tags-input button::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.cell-tag:-ms-input-placeholder,
.tags-input input:-ms-input-placeholder,
.tags-input button:-ms-input-placeholder {
  color: #999;
}
.cell-tag::-webkit-input-placeholder,
.tags-input input::-webkit-input-placeholder,
.tags-input button::-webkit-input-placeholder {
  color: #999;
}
.cell-tag::-ms-expand,
.tags-input input::-ms-expand,
.tags-input button::-ms-expand {
  border: 0;
  background-color: transparent;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
.cell-tag[readonly],
.tags-input input[readonly],
.tags-input button[readonly],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  background-color: #eeeeee;
  opacity: 1;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  cursor: not-allowed;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button {
  height: auto;
}
select.cell-tag,
select.tags-input input,
select.tags-input button {
  height: 30px;
  line-height: 30px;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button,
select[multiple].cell-tag,
select[multiple].tags-input input,
select[multiple].tags-input button {
  height: auto;
}
.cell-tag,
.tags-input button {
  padding: 0px 4px;
}
.cell-tag {
  background-color: #fff;
  white-space: nowrap;
}
.tags-input input[type=text]:focus {
  outline: none;
  box-shadow: none;
  border-color: #ccc;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
[dir="rtl"] #kernel_logo_widget {
  float: left !important;
  float: left;
}
.modal .modal-body .move-path {
  display: flex;
  flex-direction: row;
  justify-content: space;
  align-items: center;
}
.modal .modal-body .move-path .server-root {
  padding-right: 20px;
}
.modal .modal-body .move-path .path-input {
  flex: 1;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
[dir="rtl"] #menubar .navbar-toggle {
  float: right;
}
[dir="rtl"] #menubar .navbar-collapse {
  clear: right;
}
[dir="rtl"] #menubar .navbar-nav {
  float: right;
}
[dir="rtl"] #menubar .nav {
  padding-right: 0px;
}
[dir="rtl"] #menubar .navbar-nav > li {
  float: right;
}
[dir="rtl"] #menubar .navbar-right {
  float: left !important;
}
[dir="rtl"] ul.dropdown-menu {
  text-align: right;
  left: auto;
}
[dir="rtl"] ul#new-menu.dropdown-menu {
  right: auto;
  left: 0;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
[dir="rtl"] i.menu-icon.pull-right {
  float: left !important;
  float: left;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
[dir="rtl"] ul#help_menu li a {
  padding-left: 2.2em;
}
[dir="rtl"] ul#help_menu li a i {
  margin-right: 0;
  margin-left: -1.2em;
}
[dir="rtl"] ul#help_menu li a i.pull-right {
  float: left !important;
  float: left;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
[dir="rtl"] .dropdown-submenu > .dropdown-menu {
  right: 100%;
  margin-right: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.fa-pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.fa-pull-right {
  margin-left: .3em;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
[dir="rtl"] .dropdown-submenu > a:after {
  float: left;
  content: "\f0d9";
  margin-right: 0;
  margin-left: -10px;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
[dir="rtl"] #notification_area {
  float: left !important;
  float: left;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] .indicator_area {
  float: left !important;
  float: left;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
[dir="rtl"] #kernel_indicator {
  float: left !important;
  float: left;
  border-left: 0;
  border-right: 1px solid;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] #modal_indicator {
  float: left !important;
  float: left;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  height: 30px;
  margin-top: 4px;
  display: flex;
  justify-content: flex-start;
  align-items: baseline;
  width: 50%;
  flex: 1;
}
span.save_widget span.filename {
  height: 100%;
  line-height: 1em;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
[dir="rtl"] span.save_widget.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] span.save_widget span.filename {
  margin-left: 0;
  margin-right: 16px;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
  white-space: nowrap;
  padding: 0 5px;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
    padding: 0 0 0 5px;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
.toolbar-btn-label {
  margin-left: 6px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
[dir="rtl"] .btn-group > .btn,
.btn-group-vertical > .btn {
  float: right;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
[dir="rtl"] ul.typeahead-list i {
  margin-left: 0;
  margin-right: -10px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
ul.typeahead-list  > li > a.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .typeahead-list {
  text-align: right;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  min-width: 20px;
  color: transparent;
}
[dir="rtl"] .no-shortcut.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .command-shortcut.pull-right {
  float: left !important;
  float: left;
}
.command-shortcut:before {
  content: "(command mode)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
[dir="rtl"] .edit-shortcut.pull-right {
  float: left !important;
  float: left;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
[dir="ltr"] #find-and-replace .input-group-btn + .form-control {
  border-left: none;
}
[dir="rtl"] #find-and-replace .input-group-btn + .form-control {
  border-right: none;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Naive-Bayes-micro-project---income-prediction">Naive Bayes micro project - income prediction<a class="anchor-link" href="#Naive-Bayes-micro-project---income-prediction">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>By: Mathias Lund Ahrn</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># code in this cell from: </span>
<span class="c1"># https://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer</span>
<span class="kn">from</span> <span class="nn">IPython.display</span> <span class="k">import</span> <span class="n">HTML</span>

<span class="n">HTML</span><span class="p">(</span><span class="s1">&#39;&#39;&#39;&lt;script&gt;</span>
<span class="s1">code_show=true; </span>
<span class="s1">function code_toggle() {</span>
<span class="s1"> if (code_show){</span>
<span class="s1"> $(&#39;div.input&#39;).hide();</span>
<span class="s1"> } else {</span>
<span class="s1"> $(&#39;div.input&#39;).show();</span>
<span class="s1"> }</span>
<span class="s1"> code_show = !code_show</span>
<span class="s1">} </span>
<span class="s1">$( document ).ready(code_toggle);</span>
<span class="s1">&lt;/script&gt;</span>
<span class="s1">&lt;form action=&quot;javascript:code_toggle()&quot;&gt;&lt;input type=&quot;submit&quot; value=&quot;Click here to display/hide the code.&quot;&gt;&lt;/form&gt;&#39;&#39;&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[1]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to display/hide the code."></form>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Introduction">Introduction<a class="anchor-link" href="#Introduction">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This project will be looking into the 1994 census summary data to predict if income is lower or higher than $50,000 by using Naïve Bayes classifier.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Reading-and-preprocessing-the-data">Reading and preprocessing the data<a class="anchor-link" href="#Reading-and-preprocessing-the-data">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">from</span> <span class="nn">matplotlib</span> <span class="k">import</span> <span class="n">rcParams</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="n">sns</span><span class="o">.</span><span class="n">set</span><span class="p">()</span>
<span class="n">rcParams</span><span class="p">[</span><span class="s1">&#39;figure.figsize&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">8</span><span class="p">,</span><span class="mi">6</span>
<span class="n">sns</span><span class="o">.</span><span class="n">set_context</span><span class="p">(</span><span class="s1">&#39;talk&#39;</span><span class="p">)</span>
<span class="kn">import</span> <span class="nn">warnings</span>
<span class="n">warnings</span><span class="o">.</span><span class="n">filterwarnings</span><span class="p">(</span><span class="s1">&#39;ignore&#39;</span><span class="p">)</span>

<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span><span class="p">,</span> <span class="n">cross_val_score</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">confusion_matrix</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="k">import</span> <span class="n">GaussianNB</span>

<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegression</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="First-glance-at-the-dataset">First glance at the dataset<a class="anchor-link" href="#First-glance-at-the-dataset">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 16 columns):
usid              32561 non-null int64
age               32561 non-null int64
workclass         30725 non-null object
fnlwgt            32561 non-null int64
education         32561 non-null object
education_num     32561 non-null int64
marital_status    32561 non-null object
occupation        30718 non-null object
relationship      32561 non-null object
race              32561 non-null object
sex               32561 non-null object
capital_gain      32561 non-null int64
capital_loss      32561 non-null int64
hours_per_week    32561 non-null int64
native_country    31978 non-null object
label             32561 non-null object
dtypes: int64(7), object(9)
memory usage: 4.0+ MB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Number of rows: </span><span class="si">{}</span><span class="s2"> </span><span class="se">\n</span><span class="s2">Number of columns: </span><span class="si">{}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">df</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Number of rows: 32561 
Number of columns: 16
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Columns with missing data:</p>
<ul>
<li>workclass</li>
<li>occupation</li>
<li>native_country</li>
</ul>
<p>Most of the columns are self-explainatory except for <em>usid</em>, <em>fnlwgt</em> and <em>label</em>.</p>
<ul>
<li><em>label</em> shows whether the income is below or above $50K. </li>
<li><em>usid</em> and <em>fnlwgt</em> does not have any documentation of what it is, therefore these columns will be excluded in this project</li>
</ul>
<p>Datatypes in this dataset are string and int.</p>
<p>Int columns:</p>
<ul>
<li>age</li>
<li>education_num</li>
<li>capital_gain</li>
<li>capital_loss</li>
<li>hours_per_week</li>
</ul>
<p>String columns:</p>
<ul>
<li>workclass</li>
<li>education</li>
<li>marital_status</li>
<li>occupation</li>
<li>relationship</li>
<li>race</li>
<li>sex</li>
<li>native_country</li>
<li>label</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Data-cleaning">Data cleaning<a class="anchor-link" href="#Data-cleaning">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Data cleaning is done with the following steps:</p>
<ul>
<li>Dropping columns <code>usid</code> and <code>fnlwgt</code></li>
<li>Dropping all rows where NA values are present</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;https://raw.githubusercontent.com/grbruns/cst383/master/1994-census-summary.csv&quot;</span><span class="p">)</span>
<span class="c1"># Dropping fnlwgt and usid</span>
<span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;usid&quot;</span><span class="p">,</span> <span class="s2">&quot;fnlwgt&quot;</span><span class="p">],</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># Dropping all rows with missing data in following columns: workclass, occupation, native_country</span>
<span class="n">df</span><span class="o">.</span><span class="n">dropna</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">how</span><span class="o">=</span><span class="s1">&#39;any&#39;</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">isna</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[7]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>age               0
workclass         0
education         0
education_num     0
marital_status    0
occupation        0
relationship      0
race              0
sex               0
capital_gain      0
capital_loss      0
hours_per_week    0
native_country    0
label             0
dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
Int64Index: 30162 entries, 0 to 32560
Data columns (total 14 columns):
age               30162 non-null int64
workclass         30162 non-null object
education         30162 non-null object
education_num     30162 non-null int64
marital_status    30162 non-null object
occupation        30162 non-null object
relationship      30162 non-null object
race              30162 non-null object
sex               30162 non-null object
capital_gain      30162 non-null int64
capital_loss      30162 non-null int64
hours_per_week    30162 non-null int64
native_country    30162 non-null object
label             30162 non-null object
dtypes: int64(5), object(9)
memory usage: 3.5+ MB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Above is a verification of no NA values and both <code>usid</code> and <code>fnlwgt</code> are removed.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Data-exploration">Data exploration<a class="anchor-link" href="#Data-exploration">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="How-many-have-an-income-below-or-above-$50,000?">How many have an income below or above $50,000?<a class="anchor-link" href="#How-many-have-an-income-below-or-above-$50,000?">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cp</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Number of people with income above or below $50K&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;Income ($)&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;# of people&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">(</span><span class="n">cp</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This plot shows there are roughly 23,000 people with income below \$50,000 and roughly 7,000 people with income above \\$50,000.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="What-is-the-difference-in-income-between-male-and-female?">What is the difference in income between male and female?<a class="anchor-link" href="#What-is-the-difference-in-income-between-male-and-female?">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cp</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">],</span> <span class="n">hue</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Number of people with income above or below $50K differentiated by male and female&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;Income ($)&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;# of people&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">(</span><span class="n">cp</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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MYI0O6rxleasorKyskibuIhI7RUVQVFRUSVqDinNkK7mpDmbUllJ54kTpxU6HyIiTUZpaUeKinSnTJonXc2JiIiISIugwFdEREREWgQFviIiIiLSIijwFREREZEWQYGviIiIiLQI6tVBRERqpbKykunTJzN37lwqKioKnR3JQ6tWRbRqVUy7dh1o06ZtobMjUjAKfEVEpEaVlZVMmvQLs2fPoKSkNUVFxYXOkuShvLycOXNmM3PmNNq2bU/nzqUUFemmr7Q8CnxFqtGhwxKUlOjkkFFWVsH06bMLnQ0pgOnTJzN79gyWXLILHTp0KnR2pA4qKyuYNm0K06dPpk2btrRvv2ShsySy2CnwFalGSUkr5pZXMOaH3wqdlYJbefkutNZFQIs1d+5cSkpaK+htwoqKWtGxY2dmzZrBrFkzFfhKi6TAV6QGY374jQG3v1zobBRc/2P60nOl0kJnQwqkoqJCzRuagaKiaOtbqXe5Swul6hsRERERaREU+IqIiIhIi6DAV0RERERaBAW+IiIiItIiKPAVERERkRZBga+IiIiItAgKfEVEpNl7++23OPLIQ9lhhy3ZbbcdGTjwYqZMmQLA119/xemnn0Tfvlux5547c8kl/Zk48RcAfvzxB3baaVuuvfbKefN67rmn2Xbb3vz3v/8pSFlEpO4U+IqISLM2adIkzj//TP74xz144IHHuPzyaxg16iMGD76JCRPGc8IJR7PCCitw1133cdVVNzJ9+jSOPfZIZs6cyXLLLc8pp5zB008/zqhRI/n++3HcfPP1HHHEX1l33fULXTQRyZNeYCEiIs3ahAk/M2fOHLp160737svRvftyXHXV9ZSXl/Pkk4/RtWtXTjvt7HnpL730Sv74xx149dWX2XXX3dl11915663Xueaay1lyyU6ssYbxl7/8XwFLJCJ1pcBXRESatdVXN/r23Zmzzz6V0tKubLJJb7bYYmu222577rjjVsaM+ZYdd9x6gWnmzJnDt99+M+/7WWedz5//vB8//vgjDz30BK1a6YapSFOkwFdERJq9iy8eyP/93195550RvP/+u1x22QW88MJzVFRUsuGGG3P66ecsNE3HjkvO+/+HH75n6tRoEzxq1EfstNMfFlveRaTh6JJVRESatdGj/8vNN1/H7363CvvvfzDXXHMT5557ISNGvElpaSljxnzLsst2Y8UV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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>With this chart we can see that alot higher percentage of men are earning more than \$50K compared to women.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="What-ages-are-represented-in-the-dataset?">What ages are represented in the dataset?<a class="anchor-link" href="#What-ages-are-represented-in-the-dataset?">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dp</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">kdeplot</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Density of ages&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;Age&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Density&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">(</span><span class="n">dp</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The distribution of age in this dataset shows that most of the data are from age 22 to 50.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="What-types-of-job-generates-above-or-below-\$50K?">What types of job generates above or below \$50K?<a class="anchor-link" href="#What-types-of-job-generates-above-or-below-\$50K?">&#182;</a></h4><p>Are there any siginificant differences in type of job?</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cp</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">countplot</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">df</span><span class="p">,</span> <span class="n">x</span><span class="o">=</span><span class="s2">&quot;occupation&quot;</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s2">&quot;label&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Number of people with income below/above $50K by type of job&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;Type of job&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;# of people&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">rotation</span><span class="o">=-</span><span class="mi">90</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">(</span><span class="n">cp</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgoAAAGuCAYAAADxpC4wAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzdd5gUVdbH8W9PIAcRBUQQUfSwyhoRVERRWXNaM6uuuq5ZMWLEhCiuERdEERUMqGBcE+aw5jXt+rouR1EBBRVFBCRPeP+41dD0dM90z0z3BH6f55mnp6tu3bpVXeHUrVu3YuXl5YiIiIikUlDXBRAREZH6S4GCiIiIpKVAQURERNJSoCAiIiJpKVAQERGRtBQoiIiISFpFdV0AERHJjJkVAOcBXYFz3b2kjoska4BYVf0omNkE4FjgdHcfk2L8hsA3wFXufmXtFzFtucqBe939uHzNM1tm1hS4HTg0GnSUuz9dh0Wqlrpe12Y2HZju7gMShnUAFrn7ouj768CG7r5hNfKfABzr7rGal7Zhqcl6y0d+Gc5zT+AJoJ27L8twmtfJczlryszOBYYC7aJBC4BL3X10UrrJwGEpsvjI3XsnpFsXuB7YB2gOvAac4+5fJ6S5ErgC2NXdX0+aT3PgJaAfcLm7X52m3BPI8f5lZk2Add19Vq7mkWtmNhAYBWwE/Mvd+6dIcyXh9+ju7tOzyLta08VlU6NwrZk95u4/ZjuTNdiJwPHA/cA/gQ/rtjgN1tnAovgXM9sbeBDYOnF4DYwFXq6FfKRuDATezDRIaIjM7HTgJuBx4GtgfWAtYJSZfe/ujyUk3xx4G7gjKZu5Cfk1BaYAmwI3E4KO84B/mtmW7j6XSphZEfAoIUi4Pl2QkA9m1g14ERgBTKirctREVFP0IFAKnAN8lybp48A04Kc8FQ3ILlBoC9wC/ClHZWmMtog+T3f3hXVakgbM3Z9MGtSXcJCsrfzfBd6trfwk7wYSDrKN2WDAgcOBYwgB8vXATOAM4DEAMysGNgGudfcHKsnvz8C2wB7u/lI07XPA/wHnApemm9DMYsC9hJqIMe5+YY2WrOa6EwKehqwTsC5wc6qa+zh3/xT4NG+limTTmPEpYJCZ7Z6rwjRCTQAUJIjkhpmtA2xJ468R6g78x91L4wPcfTFwMHBWQjoDioH/VZHfkcBX8SAhym8q8Eo0rjJ/J1ww3ksIUqTmmkSf9fJckU2NwmBC5D7GzLaorJov1T3lVMOj788A/wYuIDTQ+Qw4nRAp/x3Ym1AtNgG4zN3LkvK8JErfDngPuNDdP0hKsx9wCbAVsAx4FbjY3b9ISFMODCccdPYEvgK2SNdYyMwOBC4kVH8vI9xaGBpFfPH8EvN+I3l9JI2/jFDtdCbQmnCFe4G7/zvbZcmkfNnON0WZMypHQvoCYA7wlrsflDD8JsIVzCHu/nhC2p+ASe5+WuJ2k9BmBuAbM1ttvZrZHsC1QK9ofncSrq5W226SyjaBhHuo0fftCVduNwLbEXbgSYTta0nCtJ2BqwlXV60JB+hrEmtBoqrR4cBeURoHRrv7uKQy9AZOjua5FfA9cBXhankY4TZWE8J94dMSq4fNbDPgGmDXKM0nwDB3fyHdcietg/2B64CNgS+Av7n7xKQ01ZqHmf0+WkcDgKbAf4Dr4uvIzJ4EdgbWif9OUXmeAka5++CEvJ4ENnX3zaJBAwlV6v9OSHMoYXveinDvfRbwCOH4sdpxK8Plrqr8F0Z5bOvuHydN+w3wjbvvVpN1SKiK7hvdMljJ3f+ZlG7z6PPzaH6t3P23FPltC6Sa58fAnmbWzt3nJY+M7nWfQVifJ7h7xi8LMrMdCMf03wPfAre5+8ho3J7A86RoC2dmkwjbR5fEQCkadxwwPvo63szGAz2BqcAN7n5BUvq/EW5ldiIEWEMJ62ws0IdwzLgLGJE4LzNrR9gHDwbWIdz+uQP4e1XrwMzaE7afA6Npp0dlvsHdSxPaDwBcYWYp24REecXTrmxrUFX+SVn0NLN7gB0Ix9gJwNXuvqKyZci4RsHdZxBW1KbARZlOl4EDo3zvIhwUexKq0V4Gygj3zT4jnJSOSZr20Gj8HVEevwNeN7P4zhLfkJ4iVNVdQLgftwPwvpklV1edA7QgBEXjKgkSTgeeJETul0R59gXeMbPtomTHAG8m/H9NFevhxKh8Ywknui0J9wst22XJsHwZzzfF8mdUjkTRCeBFYJcoEIgbEH0mNtzZDlgbeDZFVmMJDdcg/F6J67UTYdt5lXAwmEHYgQaTvQ5ReacSDihvE04+V8UTmNnawPvAIEI7lPOBJcDjUaCGmXUHPiBs5+OAIcAvwJ1mdn3SPNcjBM5vErbrEuAewnrYLVqWBwnVzzcmlOP3hABvM8JveCnht3/OzI7IYFk7Ee43vxaVbynwQPQ712ge0fb2HmH7u4mwPTYBnoi2U4DnCIH+VgmTDog+V24XUbX6bqy+XQwEXo0frM3sr4ST2K+EQPl8wnYwhIrHrUyWO5PyPwiUE36XxGXvC2wITIy+1+R3uh3oRgj4d6wkXa/o82QzmwssNLPvzSwx2GpFuJWcquHf99HnBskjzOwMwknqA0LD7OSTUFVeIuxP5xFOZrdEJz4Ix/s5VFyHLYH9gMlp5vdPwrqEcFFwjLs7IeBJ1aDzcOD5hCAoFs17CeFY9hFhP1sZrERl+CfhOH4v4djyGTASWK0habIowHgHOIGwrZ1DuJgYwarbZY9HwyEc246h6hqhbPJP9CjhuH0+Ybu+jHBcqlS2/SjcDPwXuMjMemQ5bTrrA3u6+/Xufh1hh+gCfObuR7r7nYQobjmwR9K0zYAB7n51NG1/wo43DMDM2gC3Eq5M/+Duo6NGN9sQduy/JeVXAhzq7mPd/ZZUhY2it+uBfwH93P0Wdx9GOJDEiDac6P7g1/H/E6v40ugC7O7uV7n7CGAXwtXQldksS6bly3S+KZY/23WaaAqhbcE2UV5rEU4OswhXDHF7Eg7aryZnELUniNeKPJm0XpsCf3b3C9z9DsJV/gLC9pOtdoSW3Ce7+zh3P5iwAx6VkOZCwvr7Q8I89yD87vF7vCOA9oTt9BJ3HwXsTggIzk8MagnB0WVRXmMIB6QCQnC+s7vf5u5nEIKWxH1hFOHqYJtoP7qFcDJ5G7jVQovwyjQltHY/Iypff8IB/bqo0VpN5jGKEPBvF+2n8ek+Bm6Ibh1MidLuljDdroTtYgszaxsN24FQI5MYKOzO6rcdziOcjA9y9zvc/e+EYOI74JBqLnel5Xf3bwnBXfKJ6QhCbdtjCXlV93e6kRDo9CQE9xeZ2YtmtldSuvj2tBlwGqtOOrea2dBoXOvoc3GK+cRry1omDR9EqA0oJ9QIbFxJWdMZ4+7HuPtthH381Wg52kdBwCSgv5l1SpjmAMLFW8o2KB6e0IgfA971Ve0yJgIbmlmfeNqoRmPDpLwKCI3M94mOZYdE055oZr+L0gwh7IP9o334dnc/jLBvn2ZmW1ayzBdG0x7h7me5+5ioRnUMcLiZ7R3V8sZrID+NzheZPjRQZf5J6Z919/2jY8nhhBqFY6MgNq2sAoWoeuJUwg52WzbTVuIrd/+/hO/xquv4VSMeHoGbQ7jiSvR84rTuPo1w0NnTzAqBPwBtgCfNbJ34HyEgeDVKl3j75X13/7WK8u5O2HBvcvflCfOeTriq7GNmyeXMxIuJ1ZYe7hdOAfaNrsAzXZZsy1fVfJNlu04TvUA40MRPCLsQDsKjgS3NLH4A24twlbikYhaVWkyo6Ygvy0LCgb9T2ikqNznp+3+Ajgnf9yM8cvZ2wjyXEgKUQ6NtcF/ghaR1XEaoCYkRDoSJnkj4P74vTPHVq8y/IdoXosBwF8JVefOE32OtKK+OhBqayvxKuBqLl29Z9L0j0Lu68zCzjoQA9X53/y4h/6XADYSA9A/Rifa/RNtFdJW0JeGKrYDQsh7CdjGfcGLFzDYhHPgTA4UtCAf9xOrgDsA8oFWWy51R+aPBE4GNzGzbqGwxQuDwrLv/WtPfyd3L3f1vhAurGwm3ZvsBU8zs2ISkkwm1HgPdfVJ04hxIuOocGs0zvl9XVmWefKvuJMJ6P5BwgXZvtH1nY2UNWrQPjCacS+Lr8MGobIcmTHMk8LW7v5/lvB4mLENiDcWRwG9A8iPq1yVtLzcT9s39ou+HEGoQvk865sVP7vuR3gHA/7xig+z4UyIHUTPZ5n9D0ve/R5/7VjaTrHtmdPc3CVHIHmZWVaOXTCRHTvHq/jlJw0upWN6pKfL7ihANr8uqqPdhQiSf+HcIYUdfN2Ha5Hmm0j369BTj4tVF3TLIJ9nnKYZ9SYj+25P5smRbvqrmmyzbdbqSu88hVO3FA4VdCVdmLwCFQL+olqEPqW87VGWuV6yeXMKqhkLZSn4EaRmhnHEbEtbVatz9C3efSbhf2IrstpXE/aGyfSH+THr89ziTir/HzdG4CtXISb7yirfZvoo+N6zBPDaMPjNZ/imEq8kiwgm1nFAlOo9VtU17EgLb+P3UgYSTyDfxTKNxvc3sbjN728x+JNRM/J6Kx4+qljub8j9CqPWM1yrsRKhtird3qI3fCQ/tDf5LOKluStj+bo5uy+DuD7v7CE9okxP9P45wUt6BVQ3mmqeYRXxYcqO6T4D9PPQDczdhH83maYe57v5z0rDEdY27v0eojTsMIKpJ2hN4KIv5EOU1G3iDKOiILnoOA/7hoRFoouRjYHyf3jD63JhQC5r8u8WDl8p+t+6k2H7c/QdCoFqdc0VN8k8+Z672G6RT3Z4ZLyBEMjcTovxMpYpA0/UslkkjmVRp4geD0oT5nUS4CkslscFOJvfcKus0JD7v5ZWkSSfVNPHyZ7Ms2ZavqvmmG5fpOk02BTg3OrDtSggS/kPYqPsTTqyFhCuvbKVtsFgdXkkDyEhhFfPMeltJceKCyveF+O9xG6uucJL9t5Lp0+Wfaj/Kdh7ZLP8Uwn3TPkQBpLvPN7M3CQHEuoSGuaMS8hhI0tMOZjaCUEX/CeEWxP2Eq+nRVDygV7XcGZff3eeZ2fOEk9FFhNsO81kV8Nbod4rauiyLToBE85xlZmMJNQxGuOpNJx5stnL3BWb2KxVraAE6R5+zk4af7+7z4/8Tas2uMLOnk2qE06lqXcc9CFwS1XruSQhusg4UIhOBu6K2Is0Jy5sqr+SGfMnHv0LgLRLaJyVJXleJqtqGqnOuqEn+yb9DfPpKz33VChTc/WcLLX3vInUjvVLCD7xSdKWwDqsimNqwYYphmxB20J8JDWYAfnL35APKAMIGkG0nLfE8exJOcKtlG32m6yyjMqnu+W1CiMR/sdDyH6pelmzLV+l8U4zLtBzpPEdoQLMX4SrvEncvi04IOxOqiT/3avQeVgdmAhXa6kRVwTsRWocvIvwWFZJFn9/WsAzTo8+SFL/HZoQrjlT3ohNtYGaxpOrXTaLPr1jVwC3becTLlsnyv0m4it2NsB3E5/MG4V7wgdH3KdF8CwgBxckJZelGOEnf7+5/TipnqttPVS13/ASQ6e83EZhkZlsRatceS7hlND36zPp3iu6B/5tQdZ/uKr5J1MbhPeBDdz8paXx8GeLB/SdEbYWSbA1M84pPPCTWUPxqZqcRbpncZ2Z9vIpW88DaZtbaV39UPHFdx00kPImwHyEY+dTdqwp003mMEJjF2znMJTRQTrYRq9cqxMsVr1mYDrRO8bu1I9zqrVCrmGA6KbafaHtsQ+3s/9nkvyGrB6TxxueVnpdr8lKoewj3rFLdn/kBMAtdfMYdQLi3VZv2NrP141/MrBchCn0q2vlfIjSKGxKvmovSrQ/8g4r3pjIRz/PcxMZHZtYFOJrQ9WYmtzCSHRAd6JKX5fGk+Va1LNmWr6r5JqvpOv0XYYe9jBDdvhUNf4NwNbk3Vd92iEe/df1Ss+eA7eL3pWFly/whQO/oJDGFcJtum4Q0McIBv5zq3WJZyd2/JzTGOs7Co5qJ5biH0Mq5qguCDqw6EWNmLQhtkWYA/67uPKLqzw+Bo6PtLz5dE8IjscuIGqJFJ5pXgD8S2hm8ESV/g3Dr6GLCCTB+a6Y34f5+YoPXtaPP1aqSzWwfwsE/uYxVLXfG5Y88TQh2ria0i1n5mGUNf6f/Eaq6j45uzcWnLSI0MlwA/Ddqk7SE0N/NBgnp2hKe3JlG2P8gnER7Wug2OJ6uJ+HE93CacqwU3RN/lNAY+bKq0hP21ROSyn42oc3AKwn5TiUEMQdFZcmkNiHl8SBqb/YcIeDYB3g0TUBzZtL3+BNH8fZOTxHaUCXfxx9KuOXUi/SeJqzn5LYC8Sdwnqlk2kxkm/+JSd/PIxyHnqIS1X4plLuXm9mphHvMyfk8RKgifN7MHiBcdZ1E2AFr01LgTTP7O6FdwjmEau+hURl/ttDPws3Au1FZign9LjQjVKFlxd3nJuT5tplNJNzPP42woVbnUTyITppmNopwYDybcHC4IptlqUb5Kp1viuWv0TqNag9eJBzgPk6oznw9mv/6VH3yjLcdGGJmU9y90o08h0YQqppfjdbfbMJy/Y4QbEHYYXcjPLY7inB1/sdo2M3unqqNSLYGE06YH5nZGEIgNojQEO9ir6I7XsI+c5+ZjYym/Quhmv6ghNsv1Z1HfLoPoukWEgLWbYHBSY2HpxAefy1jVQD5CaGGcCPgvoS0A4FPkub7OaGW5xIza0aoOesDHEc4VrRmddksd5Xld/clZvY4oZ+P2YRtOtW6yGoduvtyM7uU0NDybULviW0Ij3VuS3g5VLzm4uwoTXyfhnDs7Uh4uiy+XHcRarweNbMbCLUZ5xPac4xMVY4UziBsxxeb2T/c/aNK0i4GhkUBzDRCw8IdCf2BzE9K+yCh0V05mQUK8ePB0VEQfm/CLbwHCSdzqHiSjDsuCqbeItR07k/o2yJ+vhpBqCF63MzuIFyR70R4omQKq57aSSU+7SQzu53QQHl3wpNYj7t7ZdNmItv8j7Lw5Nq/CMHT/oQuuKdVNpMaXZFF96ZSbVRjiDqFIAQMAwgHx8ruoVXHnYQN6VLCFcc7wI5RQ7J4GW8htHwtITxvexFhZe7m7m9UyDEDUZ5HEDbkEYSd8x2gbzVa58ZNJjQ4uoAQ5b0CbB9diWS1LFmWr8r5pln+mqzT+Mab2FnMvwknhJWt2ivxMKFq+ngqfxwzp6Kr2+0JUf0pUVlihJb8L0dpviKcCJ6L0lxPuBI+wd3Pq6VyvEtoAf8h4Te8gRA4H+fhseGqfE44mQyKlmE5sK+7r7waqe48Eqb7iHAiGk44aR/k4ZHERPHt4tP4CTg6scWDhsR2KxXaJ0Qny30IbRPOIty73zb6/0KgTWLtT5bLnUn5YVUtwsPJbVxq8jt56JzrKEKNwaGEQLQTcKonPMrtobO53Qgn4yuBywnV0wOSjhHLCCeUKYR9/zJC47xdMwgs43n8SKhZKSIEXE0rST6PcA4YSHgVQHvgaHe/PUXahwjB4rsJJ+vKyjGVcJ7pTTgfJTbge4ZQ4/Idq/q1SfZHwrnqZqKLWndfeaEU3YLdgdCI/zDCkwLbE2qODk3+nZPKFp/2PkJwdDPhQmIISX1GVEc18t87Gj+S8CjtOZ5BF9xVvj1Scs/q6O2MdTVfEak+Mzse6OZ5fFtvPkUNGb8DzkgTSGSTV1PCk0Rjk0+IVsM3KtYFM7uKEPx1S7wgzrVq33oQEZE68QnpnzhqDE4itP+osq1EBo4k9EI5oRbyqg/aRJ+puuXOGQUKIiINiFfxHpaGysKjrb0It49uS/HkRTZ5nUe4zbM38LS7Z9Qlcn1loWv8QYTA51tP/URaztR1q3EREREIfajsRuhn4uIa5lVIaMfxHukbMTYkfQjtDuYTXhGeV2qjICIiImnp1kPdKCHU5iyo64KIiDQgbQhPROjclUeqUagbZeXl5TGtehGRzMViEIvFytFt87xSVFY3FpSX03bu3Lw2XBURadDat29FLKaa2HxTVCYiIiJpKVAQERGRtBQoiIiISFoKFERERCQtBQoiIiKSlp56EJFGrby8jCVLFrF06RL0OHj9E4tBUVETmjdvQXFxZS+glLqiGgURadTmz5/LggW/UFpaUtdFkRTKyspYvHghc+f+wKJFevKxPlKNQh1q2bIpRUWZxWolJWUsWrQsxyUSaVyWL1/K0qWLadmyLa1atSUWi9V1kSSFsrJS5s//hYUL59GkSVPVLNQzChTqUFFRAStKy5gxu/KXpHXr3I7iDAMKEVllyZJFxGIFtGrVRkFCPVZQUEjbtmszZ84Sli5drEChnlGgUMdmzJ7H8LEvV5pm6MkD6dG1fZ5KJNJ4lJWVUlhYRCymQLu+KygopKiomBUrVtR1USSJ9h4RabTKysoVJDQgsViBGpzWQ9qDREREJC0FCiIiIpKWAgURkTqw0069ee65p3OWPld5yJpHjRlFRKqhoCBGJg9SlJeHthIiDZUCBRGRaojFQhCwbEX6jpyaFhdlFEyI1GcKFEREqmnZihK+/2lh2vHrrduaZk2qPsyWlZUxceK9PPfc0/zww/cUFzfh97/fknPPvYD11++yMt2MGdM59dQTmDr1c9ZfvwuDB59Hnz7brxz/5puvc889dzJjxnTWXbcDAwfuybHHnkCTJk1qtqCyRmvQgYKZxYCzgNOArsAXwN/c/cGENHsA1wCbAz8Co939pqR8egM3Ar2BBcAE4Ap3X5GQZhPgZqA/UAI8Alzg7umPElKvqWdMqS8eeeQhJk68j6FDr6JHj02YNes7rr/+GkaPvoURI25aLd3gwedx8cWX88ILz3HeeWcybtx99Oz5O9599y2uuOISBg8+l+22255Zs77jlluuZ+bMGVx99XV1uHTS0DX0xowXE07w9wL7AS8BE83scAAz2xF4BpgKHAxMBG4ws/PjGZhZD+AVYAlwOHATcC5wS0KadsCrQEfgz9F8jwQeyu3iSS7Fe8ac9u3cSv9WlJZlHFCIVMf663dl6NAr2WmnnenUaT223XY7dtvtD3z11bTV0v3xj4dy0EGHsMEG3TjxxFPp1ev3TJ4crovuu288++13IAcddCjrr9+FPn22Z8iQS3jttZf5/vvZdbFY0kg02BoFMysGzgdud/drosGvRLUDZwKTgWHAx+5+TDT++Wi6S81slLsvAy4C5gMHuvty4DkzWwyMMrMR7j4LOB1oB2zl7nOj+X8Xpe3r7u/nZ6mltqlnTKkPdtppZ/7738+46647mDlzBjNnzuCbb75i3XU7rJZuiy22Wu37Zpv14qOPPgTgyy+dqVM/Z8qUZ1aOj3deNH36N6y3XuccL4U0Vg02UABKgV2AuUnDlwPtzKwZsDNwadL4R4ELgB2B14A9gKejICExzZho3Pjo8414kBB5EVgI7AMoUBCRarv//gmMHz+OffbZn2233Y7DD/8Tb731Bi+//MJq6QoKVq/ZKi0to7i4GAhPVvzpT39m7733q5B/+/br5K7w0ug12EDB3cuA/4OVbRU6AMcDA4GTgY2AYsCTJo3X5ZmZvU9o27BaGnf/ycwWABYN6gk8kJSm1My+SUgjIlIt998/nr/85USOPvq4lcMeeui+Ct0Zu0+lf/8BK7//3//9hx49NgFgo402ZubMGXTp0nXl+E8++YjJkx/i/PMvonnz5jldBmm8GmygkORgQi0AwLOEk3q8ji75BefxxodtgLZp0sTTtYn+b5tBmqzEYlBUVJhx+qKiQtq21Y5em7T+G7+FCwspKSmlsDAXzyhmmmeMwjSbWkEBFBbG6NixIx988D4777wLBQUFPP/8c7zxxmusvXb71co+adJEunTpQq9ev+fJJx/n66+nMWzYNRQWxjjmmGMZOvQiJkwYxx/+sCc//vgj1147jM6dO9Ohw7oV5lkfFRRUvq/pUdO60VhaaH1MuA1xJtCPECzEN6l0PZ2UVZEmFqWJ/19VGhGRarniimEsW7aU448/mlNPPZGvvprGBRdcwrx5v6zWEPEvfzmRRx55mGOOOZKPPvqQG2+8lQ026AbAbrsN5OqrR/DGG69z9NFHcNVVQ+nbdweuu+6mdLMVyUijqFFw92+Ab4B/RrcM7mVVEJB8xR//Pp9VtQSpagVaRWniaVOlaQ1Mr06Zy8uhpKQ04/QlJaXMn7+kOrOSNLKpIdD6b5iWLw/7WGlp7feMmK6WoKLylPN/663QCLG0tJwePXpyxx3jK6Q54ICDV6aJpx806M+rpUnMe8CAgQwYMLBCPvE0ifOsj8rKwm+Wbl9r376VahXqQIOtUTCztc3sGDNLbsr7cfTZndDgsUfS+Ph3d/ffgFnJacysAyEwiLdd8BRpCqN5JLeBEBERaTQabKBAKPu9hIaLifaIPj8A/gkcHDV2jDuEUEPwYfT9RWB/M2uSlKYUeD0hza5mtnbSfFoBlT9bJyIi0oA12FsP7v6zmY0BLor6PfgQ2InQGdJd7u5mNpxwIn/YzCYQHokcAlzk7oujrK4HBhH6RBgJbApcC9zp7jOjNGMI7R9eMbNhQPtouinu/k4eFldERKRONOQaBYBzgMuAvxAaMB4DXEFUy+DurxJqB34HPAkcBQxx9+vjGbj7VFbVDjxK6JXxZkLX0PE0PwO7EvpsmEjoEnoycEROl05ERKSONdgaBYDoXQzXR3/p0jwBPFFFPm8C21eR5jNCHw0iIiJrjIZeoyAiIiI5pEBBRERE0lKgICIiImkpUBAREZG0FCiIiIhIWg36qQcRkepo2bIpRUU1u06KxWK0LC+ndatmadM0aVJEQSxW4S2QcSUlZSxatKxG5agNRxxxELNmfVdh+DPPvMxaa60FwNSpnzN69Ejc/0eLFi3ZZ5/9OeGEkykqCqeR5557mmuvvYrHH3+WDh06rpbPXXfdwYQJd3HwwYdxzjkXEFM/zA2KAgURWeMUFRWworSMGbPn1VkZunVuR3ENg5V0ysrKKCjILO/Fixcze/YsTjnlDLbaatvVxrVq1QqA7777lrPOOpVevbZk2LARTJ8+nXHjxrBo0W+ce+6FleZ/zz13MmHCXRx++CAGDz6vegskdUqBgoiskWbMnsfwsXXXA/vQkwfSo2v7Ws1z5swZPPbYJH77bQ1rSk8AACAASURBVCGXXXZ1RtN89dWXlJeX07//ALp12zBlmgcemEDLlq247rqbKC4uZocddqJZs2aMHHkDxxxzPOuu2yHldPfeezf33HMngwYdw+mnn5UyjdR/ChRERBqw8vJyPvjgfR555CHee+8dOnXqzIknnsLdd49l/Phxaaf7+9/vYJttevPll1/QpElTunTpmjbtv/71Hv367UxxcfHKYQMG7M5NN13Hv/71Hvvue0CFae6/fwLjxt3OMcccz8knn16zhZQ6pUBBRKQBWrp0Kc8//wyPPjqJGTOm07t3H0aMuJEdd+xPQUEBc+b8SN++O6advnv37gBMm/YFbdu25corL+WDD96jtLSUHXfcicGDz6N9+3VYunQpc+b8yAYbdFtt+nbt2tGyZUtmzpxRIe8HH7yPsWNHK0hoJBQoiIg0ML/8MpejjjqM0tJS9tprH6655oYKtw06dOhYoVFhKtOmfckvv8yle/eNOPTQI5gxYzp3330Hgwefwj33PMBvv/0GQMuWLStM26JFSxYtWrTasEmTHmTSpInEYjF+/fXX6i+k1BsKFEREGphYLLbyyYGCgoKUTxGUlZVRVlaWNo/CwkJisRhnn30+5eWw+ea9ANhyy63ZcMPunHbaX3nhhSnsuONOK+eZrLy8nIKC1YdPmjSRU089k19+mcukSQ/St+/2DBiwe7WXVeqeAgURkQamXbu1eeKJZ5ky5VkeffRhHntsMn367MChhx7O9tv3IxaLMX78uIzaKGy2Wa8K47bYYitatWrFtGlfMHDgngAsWvRbhXRLliymZctWqw075ZQzOOqoY1m2bBn/+td7/O1v19Cz5+Z06tSphkstdUWBgohIA9S0aTMOOugQDjzwYN577x0eeeQhhgw5my5dunLCCSdz4IEH069f/7TTb7BBN5YsWcKrr77Eppv2ZJNNNl05rry8nBUrVtC27Vq0aNGCddftwHffrd7Pwrx5v7Bo0aIKbRf22GPvqHxNueyyYZx00nEMGzaUUaPGUlhYWItrQPJFPTOKiDRgsViMHXbox803j+b++yez9da9efvtN1lnnXXp2XOztH8tWrSkSZMmjB49skLNw5tvvsGyZcvYeuvQr8J22/Xl7bffZMWKFSvTvP76qxQWFrL11r3Tlm3TTXty/PEn8umn/2bChLtyswIk51SjICLSSHTvvhEXXngpJSUlGaUvLCzk2GP/wujRIxk58gb69duZb775irvvHkv//ruwzTYhCDjqqGN5+eUXOf/8szj88EF8++0M7rxzDPvv/8cqbykcffRxvPvu29x33z307t2XLbfcqsbLKfmlQEFE1kjdOrdj6MkD63T+uRLvVjkTRx55NK1ateKRRx7m6aefpE2bthx44CGccMJJK9N067Yht9wymttuu5XLLruQtm3X4ogjjuKEE06uMv/CwkKGDr2K44//E8OGDWX8+Adp06ZNtZZL6oYCBZEcyeZ9AvWlz/81RUlJGcVFBTXqGTEWi1FWXs7y5emv3jN510N9sN9+B7HffgdVmmbLLbfmzjsnpB2/zz77s88++6cc16VLV1566c2aFFHqkAIFkRzJ9H0CuezzX1KrjaCssDDG0uUlfP/TwrRp1lu3Nc2aFFFamjpQEGkIFCiI5FAm7xPIRZ//IiK1RZcxIiIikpYCBREREUlLgYKIiIikpUBBRERE0lKgICIiImkpUBAREZG0FCiIiIhIWgoUREREJC11uCQia5xsutdOJxaL0bK8nNatmqVNk0kXzuq6W+o7BQoissYpKiqgoGw5S+Z8m9P5LK9kXPMOXSkqapLT+b/wwnNcffXlFYYffPBhnHvuhQCUlJQwfvw4pkx5hvnzf8Xsd5xxxtlstlmvlekPPXR/evfuw0UXXbZaPt9/P5szzjiJ5cuXc+utY9hoox45XR6pGwoURGSNtGTOt3zx8A11Nv9NjxxC004bV2vasrIyCgqqrhGZNu1LunTpytChw1Yb3r79qi7Db731JqZMeZpTTz2Tjh3XY9KkiZx99umMHz+R9dfvkjbvH374nsGDT6GkZAWjRo1lww27V2tZpP5TGwURkQbmxx9/4Nhjj+SZZ55k2bL0ty6mTfsCs5706vX71f7WW68zEGoEnnrqcc4442wOOeQIdtppZ266aRStW7fmoYfur3T+Z555CitWrGDUqDsVJDRyChRERBqYtm3XYuONN+Gmm/7GIYfsy513juHnn3+qkG7atC/ZeONN0ubz0UcfUFpayi677L5yWJMmTdhxx/68++7bKaf56ac5DB58CqWlJYwefScbbNCt5gsk9Vqd3Hows85AV2AqsAQocff68WJ2EZF6rkWLFlx++dWcccbZPPXUEzz55GM8+OB9DBiwO4cdNojNN+/Fzz//zLx5v/DFF86f/nQIs2Z9R+fO63PssSew1177AjBz5nRat25Du3btVsu/S5cu/PjjDyxbtpSmTVc11vz5558488xTWL58ObfdNo7OndfP63JL3chroGBm/YC/A1tFg/4QleEeMzvX3SfnszwiIg3Z2mu357jj/srRRx/HG2+8ymOPTebkk4/jj388jH79+gMwe/YsTjttME2aNOX5559l+PArKC0tZd99D+C3336jZcuWFfJt0SIMW7x48cpA4Zdf5jJ48CnMmvUtzZo1o6SkJH8LKnUqb7cezGw74GWgNTAyYdQvwArgQTPbO1/lERFpTAoKCojFYtH/MXr23Iy//e0WRo8ey0477UKfPttz+eVX07t3H+666w4AystZOU2i+OOcsdiqU8Q777xFWVk5t99+N4WFhVx55aWsWLEiD0smdS2fbRSGA98AWwIjgBiAu38YDfsfcEkeyyMi0qDNmzeP++67h8MPP5Arr7yUddZZlzvvnMA551zAWmutRb9+/VfWDsTtuONO/PTTHH799VdatWrFokWLKuS7eHEYlljb0Lnz+owePZZevbbg7LOH8MUXU7njjtG5XUCpF/J562EH4Gp3X2JmLRJHuPsCM7sTGJZ6UhERiVu8eDEjR97Ayy+/QNOmzTjggD9yyCGH06FDx5VpPvvsU6ZP/5r99jtotWmXLVtGYWEhrVq1YoMNurFgwXwWLFhAmzZtVqb57rvvWG+99SkuLl45bJtterPOOusCsNde+/LWW28wefKD9OmzPX377pDjJZa6lO+nHirrgqwZegpDRKRK8+f/ymeffcqZZ57LE088x6mnnrlakAAhULjuuuFMm/blymFlZWW89tor/P73W1JUVMR22/UF4PXX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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The graph shows that all type of jobs in this dataset have a salary rate below \$50K. There are two job types that stands out where there are close to equal number of people with an income below or above \\$50K, which are in <em>Exec_managerial</em> and <em>Prof_speciality</em>.</p>
<p>For <em>Armed_Forces</em> and <em>Priv_house_serv</em> there isn't a lot of data recorded so by looking specifically for those two categories we can discover that for <em>Armed_Forces</em> there are 8 people with income below \$50K and 1 person with an income above \\$50K.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;occupation&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;Armed_Forces&quot;</span><span class="p">][</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;=50K    8
&gt;50K     1
Name: label, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In the case of job type <em>Priv_house_serv</em> there are 142 people with an income below \$50K and 1 person with an income above \\$50K.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="p">[</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;occupation&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;Priv_house_serv&quot;</span><span class="p">][</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[14]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;=50K    142
&gt;50K       1
Name: label, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Does-race-have-an-impact-on-income?">Does race have an impact on income?<a class="anchor-link" href="#Does-race-have-an-impact-on-income?">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">calcIncome</span><span class="p">(</span><span class="n">x</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">x</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s2">&quot;label&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">size</span><span class="p">()</span><span class="o">/</span><span class="n">x</span><span class="p">[</span><span class="s2">&quot;race&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>


<span class="n">perc_inc</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s2">&quot;race&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">calcIncome</span><span class="p">))</span>
<span class="n">perc_inc</span><span class="o">.</span><span class="n">rename</span><span class="p">({</span><span class="mi">0</span><span class="p">:</span><span class="s1">&#39;&lt;=50K&#39;</span><span class="p">,</span> <span class="mi">1</span><span class="p">:</span><span class="s1">&#39;&gt;50K&#39;</span><span class="p">},</span> <span class="n">axis</span><span class="o">=</span><span class="s1">&#39;columns&#39;</span><span class="p">,</span> <span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">races</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;American/Indian/Eskimo&quot;</span><span class="p">,</span> <span class="s2">&quot;Asian&quot;</span><span class="p">,</span> <span class="s2">&quot;Black&quot;</span><span class="p">,</span> <span class="s2">&quot;Other&quot;</span><span class="p">,</span> <span class="s2">&quot;White&quot;</span><span class="p">]</span>
<span class="n">ax</span> <span class="o">=</span> <span class="n">perc_inc</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">stacked</span><span class="o">=</span><span class="kc">True</span><span class="p">);</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xticklabels</span><span class="p">([</span><span class="n">race</span> <span class="k">for</span> <span class="n">race</span> <span class="ow">in</span> <span class="n">races</span><span class="p">],</span> <span class="n">rotation</span><span class="o">=-</span><span class="mi">90</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_title</span><span class="p">(</span><span class="s2">&quot;Ratio of of income based on race&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_xlabel</span><span class="p">(</span><span class="s2">&quot;Race&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">set_ylabel</span><span class="p">(</span><span class="s2">&quot;Ratio of income&quot;</span><span class="p">)</span>
<span class="n">ax</span><span class="o">.</span><span class="n">axhline</span><span class="p">(</span><span class="n">y</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span> <span class="n">xmin</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span> <span class="n">xmax</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;black&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;--&#39;</span><span class="p">,</span> <span class="n">lw</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
<span class="c1">#print(embar_surv)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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Icudu2TJEoqLi+ncuTN9+qzB3LlzmDt3Liuu2HXZMT/+8C0r9+xNSZvKcTZrr7sRK3ZfCYBtd9iVd976L5OemsigDTZl/Q03a+RXLCKFohpLKzVnzmw++ugDTj75DB599GlOOOHk5ZIKhMRy5ZWXMXXqlGXbKioqePHF5xk0aANKSkrYbLMtAHjppeeXHVNaupT3332D9QZtXGMMfzz+dFbouiLjxl7JnNm/5vHViUiclFhaqZ49e3HvvRPZf/+DaN8++3Q0e+65L6us0puRI0cwefIkXnvtVc4++zSmTfuCE044BYBVVunNHnvszXXXXcP999/L6/99lasvP5cFC+az136H1hhDly5dOfaEEcyZM4u/j72KZNQuIyLNm26FNUDbXn3D6PcYn7+0nh2riouLaz1mhRVWYOzYW7nllhu58ca/sWDBfAYOXIfrrruFddddb9lxZ501ki5dunD33RNYtGgha/Rdi3MvvLpOXYk33HhLdtplb1587kmeefJh9tzn4Pq9IBFpMpRY6qmsrAJK2lGy6tpUVCRZUlpe8BhKy8IklI1plVV6M3r0mBqPadu2Laecciannz6CxUvK+fan+VWOue7m+6s9/+jjz+Do489ocKwi0jQosdTTggVh4F9xcaLaD1MRkdZIbSwiIpJXSiwiIpJXSiwiIpJXSiwiIpJXSiy1KCoqIpksfI8vqYdkkvJkOWSbu0xECkaJpRZt2rShrKyUBQvmxh2K1CSZhIpFLF68lKKS7AM+RaQw1N24Fp06daW0tJR582axaNF8EonlBxYWFUFZeZJEWbyz9M6evZCS4gQVFfHFEFdZlCcrKC8vY+GiUj76aj6/69mnoM8vIstTYqlFIpGgW7eVWLBgDqWlpVRkfHKXlBSzYPFS3p1SdfXFQtps7VVo364tS5fGd9surrIoLUuyYHEFU79fzPezStl5a90KE4mTEksdJBIJOnfulnVf164dWDBjNk/832cFjmp5W22yLquv3o05cxbFFkNTKQsRiZfaWEREJK9ir7GY2WHABUA/YDpwhbvfVcPxnYGrgAOBzsArwKnuPqW6c0REpHBirbGY2VDgXuDfwBDgJeBOMzuohtMeBIYC5wDDgNWAF82saw3niIhIgcRdY7kCeMjdT48eP2tm3YFLgYmZB5vZtsCewB7uPina9iowDfgToSYjIiIxiq3G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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>From this dataset and graph it seems that white and asian people have a higher percentage of income above \$50K compared to the others.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Preparing-data-for-Scikit-Learn">Preparing data for Scikit-Learn<a class="anchor-link" href="#Preparing-data-for-Scikit-Learn">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Before going into modelling with this data, the <code>label</code> and <code>sex</code> columns will be converted into 0 and 1.</p>
<ul>
<li><code>label</code> -&gt; &gt;50K = 1 and &lt;=50K = 0</li>
<li><code>sex</code>-&gt; Male = 1 and Female = 0</li>
</ul>
<p>All the other features that currently are non numeric / categorical, will be replaced by dummy variables for each model after selecting features.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Converting label strings into 1 = &gt;50K and 0 = &lt;=50K</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;&gt;50K&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Converting sex into 0 and 1</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s2">&quot;sex&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;Male&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For the rest of the columns of type object, <code>workclass</code>, <code>education</code>, <code>marital_status</code>, <code>occupation</code>, <code>relationship</code>, <code>race</code> and <code>native_country</code>, dummy variables will be generated.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">get_dummies</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;workclass&quot;</span><span class="p">,</span> <span class="s2">&quot;education&quot;</span><span class="p">,</span> <span class="s2">&quot;marital_status&quot;</span><span class="p">,</span> <span class="s2">&quot;occupation&quot;</span><span class="p">,</span> <span class="s2">&quot;relationship&quot;</span><span class="p">,</span> <span class="s2">&quot;race&quot;</span><span class="p">,</span> <span class="s2">&quot;native_country&quot;</span><span class="p">],</span> <span class="n">drop_first</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">shape</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[19]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>(30162, 96)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now the dataset has 103 columns consisiting of numeric data types.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="k">def</span> <span class="nf">base_acc</span><span class="p">(</span><span class="n">y_train</span><span class="p">):</span>
    <span class="k">return</span> <span class="mi">1</span><span class="o">-</span><span class="n">y_train</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>


<span class="k">def</span> <span class="nf">print_conf_mtx</span><span class="p">(</span><span class="n">y_true</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="n">classes</span><span class="o">=</span><span class="kc">None</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot; Print a confusion matrix (two classes only). &quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="ow">not</span> <span class="n">classes</span><span class="p">:</span>
        <span class="n">classes</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;neg&#39;</span><span class="p">,</span> <span class="s1">&#39;pos&#39;</span><span class="p">]</span>
    <span class="c1"># formatting</span>
    <span class="n">max_class_len</span> <span class="o">=</span> <span class="nb">max</span><span class="p">([</span><span class="nb">len</span><span class="p">(</span><span class="n">s</span><span class="p">)</span> <span class="k">for</span> <span class="n">s</span> <span class="ow">in</span> <span class="n">classes</span><span class="p">])</span>
    <span class="n">m</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="n">max_class_len</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="s1">&#39;predicted&#39;</span><span class="p">)</span><span class="o">//</span><span class="mi">2</span> <span class="o">+</span> <span class="mi">1</span><span class="p">)</span>
    <span class="n">n</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="s1">&#39;actual&#39;</span><span class="p">)</span><span class="o">+</span><span class="mi">1</span><span class="p">,</span> <span class="n">max_class_len</span><span class="p">)</span>
    <span class="n">left</span>   	<span class="o">=</span> <span class="s1">&#39;</span><span class="si">{:&lt;10s}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;10&#39;</span><span class="p">,</span><span class="nb">str</span><span class="p">(</span><span class="n">n</span><span class="p">))</span>
    <span class="n">right</span>  	<span class="o">=</span> <span class="s1">&#39;</span><span class="si">{:&gt;10s}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;10&#39;</span><span class="p">,</span><span class="nb">str</span><span class="p">(</span><span class="n">m</span><span class="p">))</span>
    <span class="n">big_center</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="si">{:^20s}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;20&#39;</span><span class="p">,</span><span class="nb">str</span><span class="p">(</span><span class="n">m</span><span class="o">*</span><span class="mi">2</span><span class="p">))</span>
    <span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">((</span><span class="n">left</span><span class="o">+</span><span class="n">big_center</span><span class="p">)</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s1">&#39;&#39;</span><span class="p">,</span> <span class="s1">&#39;predicted&#39;</span><span class="p">))</span>
    <span class="nb">print</span><span class="p">((</span><span class="n">left</span><span class="o">+</span><span class="n">right</span><span class="o">+</span><span class="n">right</span><span class="p">)</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="s1">&#39;actual&#39;</span><span class="p">,</span> <span class="n">classes</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">classes</span><span class="p">[</span><span class="mi">1</span><span class="p">]))</span>
    <span class="nb">print</span><span class="p">((</span><span class="n">left</span><span class="o">+</span><span class="n">right</span><span class="o">+</span><span class="n">right</span><span class="p">)</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">classes</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="nb">str</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">]),</span> <span class="nb">str</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">])))</span>
    <span class="nb">print</span><span class="p">((</span><span class="n">left</span><span class="o">+</span><span class="n">right</span><span class="o">+</span><span class="n">right</span><span class="p">)</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">classes</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span> <span class="nb">str</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">]),</span> <span class="nb">str</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">])))</span>
    
<span class="c1"># Function that calculates precision and recall and returns a panda series with its corresponding</span>
<span class="c1"># name and value</span>
<span class="k">def</span> <span class="nf">precisionAndRecall</span><span class="p">(</span><span class="n">true</span><span class="p">,</span> <span class="n">predicted</span><span class="p">):</span>
    <span class="n">cm</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">true</span><span class="p">,</span> <span class="n">predicted</span><span class="p">)</span>
    <span class="n">precision</span> <span class="o">=</span> <span class="n">cm</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">]</span><span class="o">/</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">]</span><span class="o">+</span><span class="n">cm</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">])</span>
    <span class="n">recall</span> <span class="o">=</span> <span class="n">cm</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">]</span><span class="o">/</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">0</span><span class="p">]</span><span class="o">+</span><span class="n">cm</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">])</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">({</span><span class="s2">&quot;precision&quot;</span><span class="p">:</span><span class="n">precision</span><span class="p">,</span> <span class="s2">&quot;recall&quot;</span><span class="p">:</span><span class="n">recall</span><span class="p">})</span>
    <span class="k">return</span> <span class="n">result</span>


<span class="k">def</span> <span class="nf">print_learningCurve</span><span class="p">(</span><span class="n">clf</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span><span class="p">):</span>
    
    <span class="n">te_errs</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">tr_errs</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">tr_sizes</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">linspace</span><span class="p">(</span><span class="mi">100</span><span class="p">,</span> <span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="mi">10</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">int</span><span class="p">)</span>
    
    <span class="k">for</span> <span class="n">tr_size</span> <span class="ow">in</span> <span class="n">tr_sizes</span><span class="p">:</span>
        <span class="n">X_train1</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[:</span><span class="n">tr_size</span><span class="p">,:]</span>
        <span class="n">y_train1</span> <span class="o">=</span> <span class="n">y_train</span><span class="p">[:</span><span class="n">tr_size</span><span class="p">]</span>
  
        <span class="c1"># train model on a subset of the training data</span>
        <span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train1</span><span class="p">,</span> <span class="n">y_train1</span><span class="p">)</span>

        <span class="c1"># error on subset of training data</span>
        <span class="n">tr_predicted</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train1</span><span class="p">)</span>
        <span class="n">err</span> <span class="o">=</span> <span class="p">(</span><span class="n">tr_predicted</span> <span class="o">!=</span> <span class="n">y_train1</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
        <span class="n">tr_errs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">err</span><span class="p">)</span>

        <span class="c1"># error on all test data</span>
        <span class="n">te_predicted</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
        <span class="n">err</span> <span class="o">=</span> <span class="p">(</span><span class="n">te_predicted</span> <span class="o">!=</span> <span class="n">y_test</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
        <span class="n">te_errs</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">err</span><span class="p">)</span>
        

    <span class="c1">#</span>
    <span class="c1"># plot the learning curve here</span>
    <span class="c1">#</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Errors by training set size&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;Training set size&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;Classification error&quot;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">tr_sizes</span><span class="p">,</span> <span class="n">te_errs</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">tr_sizes</span><span class="p">,</span> <span class="n">tr_errs</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">([</span><span class="s2">&quot;Test Error&quot;</span><span class="p">,</span> <span class="s2">&quot;Training Error&quot;</span><span class="p">])</span>


<span class="k">def</span> <span class="nf">getBestFeaturesCV</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">):</span>
    <span class="n">remaining</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]))</span>
    <span class="n">selected</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">n</span> <span class="o">=</span> <span class="mi">23</span>
    <span class="k">while</span> <span class="nb">len</span><span class="p">(</span><span class="n">selected</span><span class="p">)</span> <span class="o">&lt;</span> <span class="n">n</span><span class="p">:</span>
        <span class="c1"># find the single features that works best in conjunction</span>
        <span class="c1"># with the already selected features</span>
        <span class="n">accuracy_max</span> <span class="o">=</span> <span class="o">-</span><span class="mf">1e7</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">remaining</span><span class="p">:</span>
            <span class="n">selected</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">i</span><span class="p">)</span>
            <span class="n">scores</span> <span class="o">=</span> <span class="n">cross_val_score</span><span class="p">(</span><span class="n">GaussianNB</span><span class="p">(),</span> <span class="n">X_train</span><span class="p">[:,</span><span class="n">selected</span><span class="p">],</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
            <span class="n">accuracy</span> <span class="o">=</span> <span class="n">scores</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
            <span class="n">selected</span><span class="o">.</span><span class="n">pop</span><span class="p">()</span>
            <span class="k">if</span><span class="p">(</span><span class="n">accuracy</span> <span class="o">&gt;</span> <span class="n">accuracy_max</span><span class="p">):</span>
                <span class="n">accuracy_max</span> <span class="o">=</span> <span class="n">accuracy</span>
                <span class="n">i_max</span> <span class="o">=</span> <span class="n">i</span>
        <span class="n">remaining</span><span class="o">.</span><span class="n">remove</span><span class="p">(</span><span class="n">i_max</span><span class="p">)</span>
        <span class="n">selected</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">i_max</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;num features: </span><span class="si">{}</span><span class="s1">; accuracy: </span><span class="si">{:.2f}</span><span class="s1">; &#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">selected</span><span class="p">),</span> <span class="n">accuracy_max</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">selected</span>

<span class="k">def</span> <span class="nf">getBestFeaturesLogCV</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">):</span>
    <span class="n">remaining</span> <span class="o">=</span> <span class="nb">list</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]))</span>
    <span class="n">selected</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">n</span> <span class="o">=</span> <span class="mi">10</span>
    <span class="k">while</span> <span class="nb">len</span><span class="p">(</span><span class="n">selected</span><span class="p">)</span> <span class="o">&lt;</span> <span class="n">n</span><span class="p">:</span>
        <span class="c1"># find the single features that works best in conjunction</span>
        <span class="c1"># with the already selected features</span>
        <span class="n">accuracy_max</span> <span class="o">=</span> <span class="o">-</span><span class="mf">1e7</span>
        <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">remaining</span><span class="p">:</span>
            <span class="n">selected</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">i</span><span class="p">)</span>
            <span class="n">scores</span> <span class="o">=</span> <span class="n">cross_val_score</span><span class="p">(</span><span class="n">LogisticRegression</span><span class="p">(),</span> <span class="n">X_train</span><span class="p">[:,</span><span class="n">selected</span><span class="p">],</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">scoring</span><span class="o">=</span><span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>
            <span class="n">accuracy</span> <span class="o">=</span> <span class="n">scores</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
            <span class="n">selected</span><span class="o">.</span><span class="n">pop</span><span class="p">()</span>
            <span class="k">if</span><span class="p">(</span><span class="n">accuracy</span> <span class="o">&gt;</span> <span class="n">accuracy_max</span><span class="p">):</span>
                <span class="n">accuracy_max</span> <span class="o">=</span> <span class="n">accuracy</span>
                <span class="n">i_max</span> <span class="o">=</span> <span class="n">i</span>
        <span class="n">remaining</span><span class="o">.</span><span class="n">remove</span><span class="p">(</span><span class="n">i_max</span><span class="p">)</span>
        <span class="n">selected</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">i_max</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s1">&#39;num features: </span><span class="si">{}</span><span class="s1">; accuracy: </span><span class="si">{:.2f}</span><span class="s1">;&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">selected</span><span class="p">),</span> <span class="n">accuracy_max</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">selected</span>

<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">confusion_matrix</span><span class="p">,</span> <span class="n">roc_curve</span><span class="p">,</span> <span class="n">roc_auc_score</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="k">import</span> <span class="n">precision_recall_curve</span>
<span class="kn">from</span> <span class="nn">inspect</span> <span class="k">import</span> <span class="n">signature</span>

<span class="k">def</span> <span class="nf">plot_roc_curve</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">):</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;orange&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;ROC&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="p">[</span><span class="mi">0</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;darkblue&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;--&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;False Positive Rate&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;True Positive Rate&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Receiver Operating Characteristic (ROC) Curve&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Model-1">Model 1<a class="anchor-link" href="#Model-1">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Features">Features<a class="anchor-link" href="#Features">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For the first model following features will be used for prediction:</p>
<ul>
<li>age</li>
<li>sex</li>
<li>race_White</li>
<li>education_Masters</li>
<li>relationship_Not_in_family</li>
<li>occupation_Tech_support</li>
<li>education_Bachelors</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">predictors</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">,</span> <span class="s2">&quot;sex&quot;</span><span class="p">,</span> <span class="s2">&quot;race_White&quot;</span><span class="p">,</span> <span class="s2">&quot;education_Masters&quot;</span><span class="p">,</span> <span class="s2">&quot;relationship_Not_in_family&quot;</span><span class="p">,</span> <span class="s2">&quot;occupation_Tech_support&quot;</span><span class="p">,</span> <span class="s2">&quot;education_Bachelors&quot;</span><span class="p">]</span>
<span class="n">df1</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">predictors</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">corr</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">predictors</span><span class="p">]</span><span class="o">.</span><span class="n">corr</span><span class="p">()</span>
<span class="n">hm</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">corr</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Heatmap of the selected features&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">(</span><span class="n">hm</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAigAAAHKCAYAAAApabCRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzdd5wU9f3H8deBGkss0RBbVFTiR2M01p8hNjSKGkvsJfbYNWosWFHAbuzRxGDFrogdC2DBjr2XT2yoxF6xR7j9/fH5LgzD3t3ucbfleD957GPZme/M9zuzezuf/bZpKhQKiIiIiNSTbrUugIiIiEieAhQRERGpOwpQREREpO4oQBEREZG6owBFRERE6o4CFBEREak7ClBEOpGZDTSzgpn1aWF9z7R+SCeXY3Yz69GZedSKmR1kZu+b2Xdmdkor6aY4B2Y2xMwaZp6FzGepZyftv1tH7tvMdmnts59Jt5yZPWlm35vZWDNr6qgy5PJZrDP2K51HAYpIF2dmKwKvAkvXuiwdzcyWAc4ExgL7A8NaSNdlz0FHMLM5gDHALjXI/iJgSeAo4Ch37/Cg0cxGAMd29H6lc81Q6wKISKdbBlig1oXoJMuk55Pc/bY20nXVc9AR5gZWBu6oQd7LAre5+5mdmEdf4LJO3L90AtWgiEgjmyk9f1XTUsi0mBG9f1KCalBE6pCZ/Ro4EViLuAg/Axzn7iNy6bYkmjaWA2YB/gtcDxzj7j+Y2UBgQEp+n5m97e49U5+XlYC9gNPT9u8Dg4CrgeOAXVPeo4B93f3TcvNNaUYDE4BzgFOBRYlmlhPc/YYyzsEywPFAH+AnwHPAKe5+c2b/a2aODXefqv9CS+cgs36ldA5WAb4ErgGOdPfvM2l+CZwEbADMDrwCnO7uV7VxDE3AMcD2wCJp/yOJpox3M+l+RpzzzYGfA28C/wb+0VqTR7nbpSacQcAWKd0bwDnuflHqI3JfSjrAzAYAi7r7WDObGeifyr8gMA64kngP/5fZ/y+I93gj4jNzLfB0G+dmF+DS9HJnM9sZ2NXdh1SQb690fv8A/AL4GngYOMLdX0p9at7K5bFWen1fMb/M/vpkl2de7wIcCvwKuNrd/2Jm3YCDgD2Iz/YnRBPjMe4+PrPPNYnP8bLENbf4OW6txk9QDYpItcxpZj/PP4Cf5ROmC/OjwK+Ji+LRxK/MO8xsm0y63Ymg4AvgcOIL9G2gH3BESnYjcEH6/0nA3zJZzQ8MBx4EDiGCiUuA24G1iS/Vq4GtiQt4JfkWLUV8ad+f0jYDw8zsz62dLDNbmegTsQpwBtE/YSbgJjPbLyU7MXdsO7awu9bOAcC9wMtp+ePpeVJnWzNbAHgMWAf4RzreT4Arzaxfa8eRyj0AuAvYD7gQ2BQYaWbd0/5nAx5I5b8s5f8icDZwXks7Lnc7M5sppdufeL8PIgKZC83sACLYOiglvynt7+NUvuHEZ+NW4IB0ro4Gbih2Zk3BxP3AdsTnZwDxvp3axrkplh3iM7gj8EAF+c5LfEZWB84F9iU+r32BW1IA8XGJPF5po1yl/DNtf1gqE8DFwN+JgOgA4m9ib+DedE4wMyP+npqIz8LhwGypfKu1oxzTFdWgiFTHzRWkPZf4Yl3B3b8BMLNziS/pc8zspvQr8hAikNm0+GvZzP5F/GLcAhjk7s+b2aPAnsAodx+dyWduYH93Py9tO5b4Ml0CsExNyHLEl35Rm/lm0i4AHOTuZ6d0FwLPA6eZ2bXu3tzKOWgGVnb3cWnb84mLwWlmdp27jzKzBVs4tknaOAcAA9z9rEz5nKiRKAYyJwEzA79x9/fTsvPM7CrgeDO7zN0/auE4tgfudPcDiwvM7F1gH6AnUZPRjzjnK7n7CynZ+WZ2EnCkmV3g7s+V2He52+0G/BbY3t2vTmW4gAgqjiQuvjcDZwHPu/uVKc0uRM3E+tmaOzN7HBgMbALcAuxOdHLdLFO7dSER7M3ZwnnB3d8E3jSzK4A325HvLsA8wGru/mom3VdEoLycuz9NBJL5PJZqqVwteNLd983k0Sflv7e7D84svwMYQdRMngP8iQhINnP3T1Kaa4FHgOWBhyosx3RFAYpIdRxKVO3mzUtUXQNgZvMQzRbnArOY2SyZtDcRI1ZWJi7UywKz5ZoAfgF8Dvy0zHLdlPn/f9LzncXgJHkL+H3mdSX5fgn8q/jC3b9LgcYZwIrAE/kCpV/GqwDnF4OTtO33ZnYa0QSzbnruCJP24+7NZvY0sFkqSzeixuM+4MdU61V0I/DnVJaWmnrGAWuZ2YHAte7+YbqgDc6k2YKo+Xg/t/+biQBiI0p/dsrdbiMi4M0eZ8HMdiRq5loKErdI2z2V2/8dwMS031uIZq8Pi8FJ2v83ZnYR8XmtVFn5uvupZnZpNjhMfy8T08ty/wbKMSL3egugQNRqZsv4NPBBKuM5xPsPEdCe5u5PpaZS68CydVkKUESq46lSv/Bt6nknFk/P+6dHKQsDD7v7j2a2kpltR/yC7UUEChBNLuX4MPP/Cek5XxswkaiiBqDCfN/I9hlIXkvPPSkRoKTlEDUZecXq+UVKrGuv/PF+R1y4IfprzEkEKZu2sP3Crez7UOA2otnlLDN7imgiuNDdP0hpFif68Xxc4f7L3a4n8T5M0ZfF3Se9V9ESUXL/Pcrc/5sl1r9aYlk5ys0XYCYzO4EIdnsRfUG6p3Ud2YUh/xlZnPibeKeF9MU+KNcTwe42wDZm9j4RaF3m7g92YPm6JAUoIvWl+OVarHYv5SUAMzuZqMp+hmhyuYKoOj6P1i+ak7j7hBKLW52HosJ888EJTD7GiSXWQSYYKqF40Sm133ZppZkJJpd1GFPWemSVujgX9/28mf0KWB/YOD0fBxxsZr1T00R3oqp/UAu7ea+VspWzXXdariVpTXcimNy3hfWfp+cC0QSW194Aoax8Lea2uR/4Frib6P/yNBE8/HMa8i4l/1ntTow82ryF9N9BBPPAVqlf2eZEbdOuwG5mdqS7tzixoChAEak3Y9PzBHe/O7vCYmTPosC3ZrYIESRc4e475dLN11mFa0e+i5pZU+7X+6/S82sl0sPkc7BkqSKk53dLrOsMHxMXwBlLvB8LAysA35TaMHX2/C0w3t1vJXWuNLOtgeuI0R+HEMc7e4n9/4zoi9HaeSpnu3eIZrl8+TYAtiU6fra0/5WAe7NBnJnNSFxsi+/Bm8AaZjZDLuBt78yt5eZ7GvADsLS7f5xJd1QZeRQDjp/klpf7tzOW6Jf1pLt/kV1hZlsAn6b/Lwws7O4PAS8AgyxGhN1L9CFSgNIKjeIRqSOpE+aTwC5p9Agw6cv5EuKX/AxEB1eI0Sdk0v2RCACyPz6KX8Yd8fdeSb4QfWy2zqSblegg+lqmY+cUUtPHk8AO6cu8uO1MwMHERWlUheVu1zlIF9w7gA3N7Le51WcSfXh+PtWGoTvRd+Xs3PLHcmW6FfitmW2YS9efaCL4TQv7L3e7O4B5zWyzXLqDgA2JEUmlzs+txPu9T267vYlhxOuk1zcSzWC7FxOkz+ueLZS7LeXmOw/wUS44mZPJs+FmP4vNTHlsxea15XJ5bEN5iiN5js4uNLONib/R4ii1o4B7UmduAFK/qnG0XIMoiWpQROpPcVjlU2l0zKfEEM5ViPk5PjWzr4lfxkelIY3jgP8jvpy/J+bqKCp+ge9jZvMVR3K008sV5AvwI3Cpma1ANDn8Bfgl0YmwNcVz8EQ6B18BOxB9DQ7I/2otw7ScgyOIYdcPmNk/iX42G6XHYHd/qdRG7v4/M/sH0N/MbiKGGs9KXLi/JQJOgJOJTpc3mtm/iSa81YghsXemRynlbjeYOO/XpvI7EZisC/zF3Sea2afERXwTM3ubCDouAnYGzk3v3+PEjLx7EU0pxTlMrkjHdF6q5fsP8V61tyav3HzvBA43s6HE3DLzEUHSvGl9/m+gj5ntAYxw99dSf6A9zeybVObNKL/W5w6ig/ChFvf4GUX0xfkr8fdRHJb/T2An4rMzmGieWpuYi0VT77dBNSgidcbdHwVWJWoRDiGqsmcDdim2WadRNn8k+oAcSHwhrpj+fzgwR2qjB7gHGEpclM5LgUV7y1ZJvhBByZ+JC+lJxKieddx9ZJnn4Cmio+kJRAC0qbuf246it/scuPsbRHB4O9EsczZxITuYmNukNQNSul7EyKUBpCaR4tBYd/8M6A0MAbYi5lr5HTEPzZYt9ZEpdzt3/46Y7O5iItA9i5j8bGt3vzSl+ZaoDViIGEH22/Re/yGV+w9p/xsB5wN90za4+0RgvbR8a6LZ4h3iYl2xcvMFBhKfv96pzLsSgcJyRLC1dma3hxMdn89l8uR+WxJBxl7EnC0fEUOYyyljgTjn/YmaqnOIoOwGYHV3/zCle4Go8Xmd+ByfS9wPan/iMy2taCoUGuZmniLSQCxmeu3pmVlbRaRrsJgf6Qli1uFxraT7KREAbkEM/X4AONDdW+pbNYlqUERERKRsFuPSh1NeN5HriNqmw4nmrgWJW060OIlfkfqgiIiISJvMbAaiv9EpRP+yttKvRjQJb+Dud6VlDxKTP+5NG7dDUA2KiIiIlGM14v5DZxA1Im3pS3RwnzTqLo26up8IXFqlGhQR6RTu3qfWZRCRqZlZm6Pg3H2uEotfARZz94/SPZPasiTweupInfU6ZQzpVoAi0gl+/OTNuux9vt2K+Rv51odPJ35X6yKUNF/32WpdhBa9+kNLM8HXVlNTaxMB184s3WaqdRFa9PB/753mk1bhd86X7cmjODqpAnMyedr/rK+AOdraWAGKiIjIdKSF2pHO0ETpW2c0UcbtFxSgiIiINLrmupyY9ktKT343O2XU4qiTrIiISKObOKH8R/U4sJiZ5ZuwelH6buVTUIAiIiLS4AqF5rIfVTQSmIvJ90/CzHoAaxB3oG6VmnhEREQaXXNVA4+SUvCxOPCyu4939wfSjNLXmtlhwGfELQq+IG5d0CrVoIiIiDS6QnP5j86zIXGfrhUyyzYn7v58OnHfqHHAH9z987Z2pnvxiHQCDTOujIYZV07DjCvT1YcZ/+/tp8v+zplpkRXq803KUROPiIhIo6tu35KqUIAiIiLS4ArVHZ1TFQpQREREGl0ddJLtaApQREREGp2aeERERKTu1OdMstNEAYqIiEijUw2KSH0xs9mAY4mx9gsDPxDj8Pu5+/Mpze5Av7T+OeAk4BZgLXcfndIsA5xKzHA4ERgBHOzu46p5PCIi7dIFO8lqojZpdFcAOxNBR1/gYGAZ4GozazKzXYELgVHApsD9wLXZHZjZEsDDwNzADsCeaR8PmNmcVToOEZH2a24u/9EgVIMiDcvMZgZmBf7q7sPS4vvNbA7gDODnxLTKw9z9r2n9CDObHdgns6sBwNfAOu7+ddr3/cCbwF+BEzv7WEREpkWhoD4oInXD3b8H1gcwswWBJdJjo5Tk10SzzuG5Ta9jygDlD8SNq743s+LfxCfAY8C6KEARkXqnPigi9cXM1gPOBpYEviL6mHydVhf/YvNzgn+Qez0PsH165L3WMSUVEelEDdR0Uy4FKNKwzGxx4GbgRmBDd38zLd+XqFl5OyX9RW7T/OsvgTuBc0pk80OHFVhEpLOoBkWkrqwIzAycVAxOkg3ScxPwFvAn4JrM+k1z+7mfaA562t2bAcysOzCUaOZ5oeOLLiLSgSb+WOsSdDgFKNLIngYmAH83s7OIYGVX4pbfEB1oBwKXmdmHwHBgVaLjK0xuAjoOGAPcamYXAD8C+xP9T87v/MMQEZlGXbCJR8OMpWG5++vAdsAiwG3A4LSqD1AAVnf3y4mAZGMiQFkHOCKl+zrt5zlgdSJgv4roRDs78Ed3v7saxyIiMk0KzeU/GoRqUKShpeHFw0qs6gZgZtsBI9z9n8UVqY9KMzGMuLifJ0kjgkREGk4XrEFRgCJd3c7AQDM7hhi9szRwAnCFu39R05KJiHQUBSgiDWcnYgr7s4nhxOOAf6C5TUSkCymok6xIY3H3j4iOsyIiXVcD9S0plwIUERGRRqcmHhEREak7qkERERGRuqMaFBEREak7qkERkXJst+Lfal2Ekq556uxaF6GkXVc8tNZFKOmNCZ/XuggteuGzsbUuQknLzrNorYtQ0oxNXXxe0gkTal2CDqcARUREpNFVqQYlTX7ZH1gMGAucnGbsbil9D+DvwHrE7UgeAQ5y9zbvFN/FQ0oREZHpQHNz+Y92MrOtiNuBjCRuujqauNfZli2kbwJuIm7gegSwIzAfcJ+Z/ayt/FSDIiIi0uiqU4NyMjDU3Q9Kr0eY2dzA8ZS+5civiBu07lysZTGzV4A3gE2Ay1rLTDUoIiIija6Ta1DMbDFgceCG3KphwJJmVqrz0czp+avMss/S8zxt5akaFBERkUZXQQ2KmbV5HzJ3nyu3aMniqtzy14u7Bd7K7eN5M7sPODbVnHwKnEHcSf7mtsqgAEVERKTRdf4onjnT8/jc8mLtyBwtbLcPMAJ4Jb3+AdjU3d9sIf0kClBEREQaXaFQdtIStSPlaCrm1MLyqapwzGwpYtTO68DfgG+BPYAbzGx9d3+wtQwVoIiIiDS6zp9J9sv0nK8pmT23PqvYmbavu38OYGajgAeBs4CVWstQnWRFREQaXecPMy72PemVW94rtz5rEeDlYnAC4O4F4CFg6bYyVIAiIiLS6ArN5T/awd1fJzrB5uc82QJ4zd3fKbUZ8JsSc578jpjkrVVq4hEREWl0EydWI5fjgEvN7HNgODGXydbAtjBp1tjFiVqT8cCZwA7EfCmnEH1QdgLWLG7TGtWgiIiINLoqzCTr7kOAvYlp628G+gA7uft1KcmGwKPACin9WGKitg+AIcC1wELAupltWqQaFBERkUbX+Z1kAXD3wcDgFtYNIQKR7LJXiJqWiilAkYZnZisSN6NaiagVfAzo7+5j0vo1gBPS+m+Je0P0c/cvzKw7MAZYGFgy09N8KNAXWLaFtlURkfpRpZsFVpOaeKShmdkcwF3AJ0RnrW2B2YC7zGyOFJzcTUwmtBXQj6iGHGFmM7j7RGAXYhKiU9M+t01p91NwIiKNoNBcKPvRKFSDIo3u18DPgXPc/REAM3sV2JMYn38y8DKwsbs3p/XPAE8D2wBXuftLZjYIONHMhgPnETfEuqrqRyMi0h5VauKpJtWgSKN7EfgYGG5m/zazzYAP3P1w4HNiONtwoJuZzWBmM6Rt3gbWzezn78CTRPPP98T0zCIijWHixPIfDUIBijQ0d/8aWB24nagRuRH42Mz+DfQgPuNHAz/mHj2BBTL7mQhcndI/4u6fISLSKKowiqfa1MQjDc/dHdgxdXj9P2BHogbkfeK+EacDQ0tsOukW4GY2P3As8CywlZlt6O63d3bZRUQ6RAMFHuVSgCINLTXpXAAs4+4fEGPwHzWz7YCfAc8AS7j7k5lt5gCuJ2pMitMzX0A07fQBrgMuMLOl3b3N25KLiNRcBTcLbBQKUKTRPUw0y9ycZiocTzT1zEE094wg+qcMISYJ+glwBLAMcAiAme0MbARs6e5fmtm+RD+Vs4kRPiIi9a0L1qCoD4o0NHf/iJiv5EvgYqIvygrAFu7+gLvfCaxP3NDqRuDSlLaPu79oZgsQgcit7n5D2uebxJTOO5vZhtU+JhGRijUXyn80CNWgSMNz96eIqZdbWj8KGNXCuveIpqD88lOAUzqqjCIinaqBRueUSwGKiIhIgyt0wSYeBSgiIiKNroGabsqlAEVERKTRdcF78ShAERERaXSqQREREZG6M0GdZEVERKTeqIlHRERE6o6aeESkHJ9O/K7WRShp1xUPrXURSrr0qdNrXYSS+i63V62L0KJVeliti9BQmmiqdRE6lYYZi4iISP1RDYqIiIjUHQUoIiIiUnc01b2IiIjUm4JqUERERKTuKEARERGRuqNRPCIiIlJ3qlSDYmbbAf2BxYCxwMnufnkr6bsBRwK7AfMDrwMnuvu1beXVrSMKLCIiIjXUXCj/0U5mthVwFTAS2BQYDVxmZlu2stnZwDHAecBGwBjgajPboK38VIMiIiLS4AoTq9LEczIw1N0PSq9HmNncwPHAsHxiM1sc2A/Y090vTovvMbMlgPWBO1vLTDUoMt0xs649paSITH86uQbFzBYDFgduyK0aBixpZouW2GxT4FtgiiYgd1/T3Q9sK08FKFITZjaHmU0ws6nmODezK82sYGYnl1h3k5mNTf8fYmavt5HPLmlfv0yvlwIe6pCDEBGpE4XmQtmPdloyPXtuefE7uNS9F5ZN6dc1s+fSd/5rZrZNORmqiUdqwt3Hm9mTwKrZ5al2Yx3gU2A9onNV1mrAzRVkdTvQG/govd4C+H17yiwiUrcqCDzM7Iu20rj7XLlFc6bn8bnlX6XnOUrspgewMHAJ0Q/lLWB34Foz+8jd72utDKpBkVq6B1jBzGbOLFsOmBc4HVjOzH5RXJFqP34OjCo3A3f/2N3HuPv/OqjMIiL1p7mCR/sUm8bzkVBxeak9z0QEKbu7+4XufjewHfAcMLCtDFWD0oWlppAbgBWAFYGLiLbAAURNxFzAh0Qb4hHu/n3abiYi2t2BCBaKw8Kuy+x7M2Ko2dLAZ0TP7v7u/kMFRbwHOApYicnNLn2BccC/gRPS6yvTutWJP457c8e5G3AEsBDwKnCYu49M63YBLk3rdk/HjpkVgEHuPjANgzuCGAb3SyLKPy3TqUtEpK4VJpQfeZSoHSnHl+k5X1Mye2591lfARGLUTzHvgpmNIr6PW6UalK7vAOApYCvgFuABYGZgZ2AD4DrgwJSu6CrgYCJI2Dhtc42ZbQRgZn8GbgReJDpBnQTsBVxdYdkeBr5nyiaX9YCR7v4F8ER6XbQ68LS7f5JZtijQDziaaL5pAm4ys3lK5HcRMDj9v3d6DXA+EbgMScc7HLjQzPav8HhERGqj82tQin1PeuWW98qtz3qNiDNmzC2fialrYqaiGpSu712gX4pa1weeAbZy96/T+rvNbF1gTeDvZvYbYEtgP3f/V0pzTxoutpaZ3Q6cCgx3952LmZjZu8DNZraquz9cTsHc/Qcze5jUD8XMZkv/LwYRI4G9zazJ3QtEgHJNbjfdgI3d/bW0j++Bu4FVgDty+Y0zs3Hp/2NS+iWAPdI5OqOYr5l1B443s4vd/dtyjkdEpFY6+1487v66mb1FXB9uyqzaAnjN3d8psdldxA/IrYmabMxsBmKI8YNt5akApet7KV3ccfe7gLvMbEYz+zUR+S4D/IJo6oFo+oEpP4C4+wYAZrYk0QxyXPqgFY0A/gesS9SMlOseoDimvg/xmbw7vR4FHAssY2afA4swdf+TD4rBSfJWei63CnNtotblttzx3Ar8Dfg/YjIiEZH6VZ2Z7o8DLk3fx8OBTYjgY1sAM+tBDEV+2d3Hu/u9ZnYH8A8z+ynwH2Bfoub7z21lpiaerq8YeGBm3czsFKLPyEvEzH7LA98xuaNTsWnkI0orrr8A+DHz+I6otlugwvLdA/Qws18R/U2edvdP07oxRI/xtYjak++YOvj5Jve6+Gda7me7eDzOlMdT7OdS6fGIiFRdFYYZ4+5DgL2JpvebiR+VO2X6J24IPEr0eyzakugucETapgewrrs/1VZ+qkGZvhxB1FbsBdzk7l8CmNnjmTTFjk49gA+KC1PTz2yZ9QdRej6RT0osa81TwBdEk8y6ZGpu3H2Cmd1HNPt8CjxQYSfcchSPZ01iQqG8t0osExGpL1W6V6C7D2ZyM3x+3RCiL1922XdEM0+/SvNSgDJ9WQ14Pn2IADCzBYlmnkfTomLQsTFwYWbbc9JzX+BjoKe7n53ZT0/gYmJ48NhyC+TuE83sfqJNcilgn1ySUUQH3u/JzUbYThNzrx9Iz3O7e/H/mNnmRC/zvYngSESkbhUm1LoEHU8ByvTlceAYMzsMeIzog3IU8BOidgR3f9bMbgTOTG2GzxMjdfoA66WAoj/wLzNrJu6lMDcwiOj38Uw7ynUPEdh8DTySWzcS+Ef6f9nzn7TiC5h0R85H3f15M7sGuCRN5fwMMXT6JOCpFjp+iYjUlUKValCqSX1Qpi8nE0NqDyICi37AFcSEOcuaWXF8+5+BfwGHALcRQ3I3SZPs4O4XEHOk9EnrzwVeAFZ390nNQhW4h+i/Mtrdf8yuSB1g3yaajl5ox77zridqiy4DDk3LdiaCoL8SnX0PI2qDNumA/EREOl/nDzOuuqZCoXOHJolMj9b65bp1+Ye1YPfZ205UA5c+NdUtmepC3+X2qnURWvS/Qr61UlozU1P3WhehRfeNGzXNNzD9eN01y/7O6THq/oa4YaqaeKTDpTlE2voDKLi7vmFFRDqAmnhEyvMGUw7ZLfW4p2alExHpYgoTm8p+NArVoEhn2JjoeNuar9pYLyIiZeqKNSgKUKTDuXtHdGYVEZEyFZobp2akXApQREREGpxqUERERKTuFAqqQREREZE6oxoUESnLfN1nq3URSnpjwue1LkJJ9TrfyMhnS95ypC7U6zn7auL3tS5CSa9/+16ti9CpmhtodE65FKCIiIg0OHWSFRERkbqjAEVERETqTle8a40CFBERkQanGhQRERGpOxpmLCIiInVnokbxiIiISL1RDYqIiIjUHfVBERERkbrTFUfxdKt1AaQyZtb1wuRpoPMhIhI1KOU+GoUClAqZWU8zK5jZDjXIuzcwvB7KkvIfkvIf20qaK1OaIR2c94JmNhxYpCP3KyLSiCY2dyv70Sgap6QCsBuwdOb1+0Bv4K7aFAeAArCImf1ffoWZzQxs0kn5rgVs2En7FhFpKIVC+Y9GoT4oDczdfwDG1LgYbwE/AbYEHs+t2wCYALxT7UKJiExPmjWKp+sxsz2Ag4DFgfeAwcCp7l5I6zcHBgBLAK8Ax+W2Hwj0d/cZcssLwDHufkJ6PT9wKnHR/gnwBNDP3Z9O63ukff8RmB/4GrgPONjd305NJDtn9r0rMJoIEHZ09yvTuqWAk4HfA7MCDwKHu/vzaX2ftN+1gaNTuvHAEOBod59Y4SksADcQAcphuXXbADcCfXLnptVjTWkWB84CVgVmAZ4Djnf3O8xsF+DStLu3zOwyd0yFr8QAACAASURBVN8lbdfW+zkEWAAYm8r3UspjHeB44DfAj8D9wBHu/mqF50NEpOqqNczYzLYD+gOLEd+jJ7v75WVuuxDwInBa8drYmum6icfMjiQuYHcBGwMXERfO09P6jYFhwPPApsBQ4Mp25PNT4GFgDeAQYCvi3N9tZr9MHT3vJIKGw4G+wEBgXeD8tJvjgVuBD4hmndtL5LMMEfjMD+wN7AT8HHjYzH6dS34NEeBsCFyd8t2l0mNLhgKLmtmKmbLMCmwEXJsrY5vHambdiL42swE7AH8CPgVuTYHL7WkbgM2Jc9Pm+5mxFtF3ZTPgJGBR4BbgybTd7sCSwO3qhCsijaAaTTxmthVwFTCSuCaOBi4zsy3L2LYJuASYo9z8ptsaFDObk4gC/+nuB6fFI83sa+AMMzsHOBZ4zN13TOtHpNqLUyrMbhegJ7Csu7+Y8h8DPE38en8Y+Ao4wN0fSduMNrNeRL8T3P0NM/sY+MHdx6R9zJbL51jgG2Btd/8mpRkJvAEMIgKjosGZCPY+M9uUCCgurvDYAB4B/kvUojyVlm2YynJfLu2CbR0r8AsiQDje3e9Mx/E4UZM1czoXb6S0z7j72HLeT3cvNjXNAOzl7mPTvrclamlOcvf30rJ3icDop6m8IiJ1q0pNPCcDQ939oPR6hJnNTfxIHNbGtvsQ3+tlm24DFKIWYlbiV3n2PNwGnE1cYFcEjsptN5TKA5TVgNeKwQmAu39JNEMUrWVmTWbWE/gV8UauCsxUQT5rALcWg5OUz9dmdisR7WY9nHs9jqixqJi7F8xsGBEAHZkWbwMMc/eJZpZNO462j/VD4GXgQjNbDxgB3JkJPEpp6/1cm2jGAvi6GJwkY4DvgSfM7Hqihme0u+f71IiI1KXOHp1jZosR16wjc6uGAVub2aLu/lYr255KXCPuLDfP6TlAmSc9j2xh/c+AJuDj3PL325nXR60lMLPtieh0IeAz4Bng21SGcs1NNAHlfQjMmVv2be51M9PW5DcUONDMlgNeJ/qXrFcqYVvHmgKedYkakc2JpqofzewmYG93/7zEbtt6PxfI/P/D7IpUA7MmcATRvHMg8IWZ/ZPoR9RA/d5FZHpUyZeUmX3RVhp3nyu3qFj74bnlrxd3S/SJzOfVjfhxONTd78r+YG3L9BygfJmetyWaQPLeI6qt5s0tnyf3ukDuwp76nOTzWiifgZmtTgQU8wKXE7/0z3T3/6b1fydqBsr1OTBfieXzA59UsJ/2eJSohdmS6Hj6GfBQPpGZrUYZx5qaWvY1s/2A36b9HkEEevuXyL+c97NFqbZkczObiajx2ovoRPwM0QlYRKRuVaGJp/gjd3xuebEJvKW+JX8jOtRuXGmG03OAMgb4HzC/u19XXJgmQxtEVGM9AmxpZidnfkXnT/J4oMnMfpmaLyAucFkPAZuZ2ZLFUSEpiLkNOAP4gQhyBrj712l9d6LjaDb4aWuEzf3AxmY2W6YPymypzKPb2HaaZJp5tgCWIqLlUkH972njWNOcKrcBG7n7E8CzwLNmtiGTA738uSjn/SwZpJjZ/kTnZUtDt+81s6eArSkRWIqI1JtKRvGUqB0pRzGD/Pd6cXlzfgOL6pITgC1St4aKTLcBirt/YmZnACelDpYPEiM7TiR+jb9E9D+5FxhmZhcSVVz5Pim3A2cCF6dagIWJzpzZjpWXAAcAt5nZAKJ2oR/R7+EiomoM4Dwzu4xoqinWHDSZ2Szu/h3wBTCvmW1AXLTzjgMeA+4xs1OJD85hREfP40qk72hDiWh5cWD1FtIU+3W0eKzEsX0NXJGGcX9ADANejgjoIM4FRK3HHe7+ahnvZ0vuBU4DbjKz84i5W/Ym3p/hrWwnIlIXpooOOl4xwMjXlMyeWw9M+uF5GXA9MCrXN7Cbmc3g7hNay3C6HmZMVOEfTjQL3ElczO4C1nL37939QWLekoWAm4A9gb9kd+Du/yH6SPQE7iD6L+xB5te6u48nOrA+BfyTGHr7fcrnfXcfTVykV0/lOJOY3GzztIvixf4C4E1iSOxU09u7+wsp7XjgCmKukE+A3tkOup1oDFHu/7r7Y6USlHOs7v4/ov/Ki8A5RCfZTYE9i/O9EKODhhN9WU5Ly1p9P1sqtLu/RIxgmoMYfn0T0ZTX191fb2k7EZF6UaCp7Ec7Ffue9Mot75VbX7QQsAqpD2HmAVGr/SNtaCo00ry3Ig1iu0U2rcs/rLETKq5lrYqZm+qzMnfks4NrXYQW9V1ur1oXoaSvJrb4W6Cm3v02P96hfnz45avT3IHk3nm3Lvs7Z+0Ph7YrPzN7E3jU3bfPLLsOWN7dl8ilnQlYtsRuniDmvLrE3Z9sLb/6/FaQmkmT6XQvI+lEjW4REakP01AzUonjgEvN7HOiBnsToq/etjBplvDFgZdTy8FUAUgaxfNeW8EJqIlHprYzU1bHtfRYs1YFFBGRKTVX8Ggvdx9C9M9bD7iZuI3JTpmBCRsSIzpXmIZsJlENiuTdBqxcRrp8e6OIiNRIlWpQcPfBxC1FSq0bwuQJMVvavuyCKkCRKbj7p8R9b0REpEFUYRRP1SlAERERaXATq1SDUk0KUERERBpcc9eLTxSgiIiINLpm1aCISDle/aE+51x44bOxtS5CSav0KP8GYtVUr3ONQP3O0fKH3+5R6yKUtPKci7edqIF1xTkfFKCIiIg0OHWSFRERkbrT3KQmHhEREakzbd3qvhEpQBEREWlwGsUjIiIidUejeERERKTuaBSPiIiI1B018YiIiEjd0TBjERERqTsTu2ANSrdaFyDPzLrgaa4POrciIl1TcwWPRtEpAYqZ9TSzgpntUOF2RwKHZl4PNLMJHV7AtstRVvnNbKyZXdQJ+Q9J+W/YwvrXzWxIhfvsDQxvR1k66xi7m9nFZjY+Pdbo6Dxy+RXMrH/6/y7p9S87M08RkWrpigFKvTXxHA+ckHl9EXBnjcpSjs2ALztx//82s6XdfXwH7Gs3YOl2bNdZx7gu8BfiPb8beKoT8sjqDbzbyXmIiNREoQvWj9dbgDIFdx8HjKt1OVri7s904u6/AeYDzgBqdvetTjzGedLzpe7+ViflMYm7j+nsPEREaqWRakbK1WaAYmZjgRuAFYAViVqNE4FTgD8BsxO/fg9394db2U8f4ChgZWA2IvAYApzg7s1mVhzGPcDMBrh7k5kNBPq7+wyZ/ewIHAQY8DlwLXCMu3+X1g8hLuzXA0cACwOvpPKNSGm6AccB2wMLAO8B1wAD3P3HTLEXNLMbgPWAH9I+D3H3bzLn5m53393MegJvAdsAuwJrAh8A57j7Oa2d4xZ8UDwGM7vW3e9pKaGZzQD8lQhkFgPeBy4GTnH3iemc7JzSFoBd3X1IOYVo4Ri3AHYA+gL/A4YBf3P3b8vc56TyAG+a2f3u3sfMehDvyx+B+YGvgfuAg9397bTtaOBl4CNgb+CnRNPVHsB+wP5p2d3Anu7+aea4j3H3bA0dZrYRcBuwtrvfl1m+PlF79xt3f6mc4xIRqZWuONV9uX1QDiCCkK2Am4F7gA2BI4EtiUDhHjNbudTGZrYCMAr4ENga2Bh4CBiUtoeogp9IXFh7t7CfQcBlwP1E08OZwF7AbbkOoL8DDgGOATYFJgA3mNmcaf3hwL4p/77A+cBhRACVdSIwFtgEODvldUypsmUMJgKezYkL59lmdkAb27RkEPAqcKGZzdZKuouBU4GhqayXAwOAC9L644FbiaCnN3B7O8tTdBHwJhGgngbsTnwWynU8MDD9f3Ng3/T+3QmsTbw/fVOadYn3J2sH4PfATsCxRFD4RNpmj7TdnzJ5tOZOIqDbMbd8J+BJBSci0giam8p/NIpym3jeBfq5e8HM9gCWBf7P3Z8EMLM7gceBk4gLSt4ywAhgJ3cvpG1GERfTNYGh7j7GzADGlaqON7O5iQvXv9z9oLR4pJmNA64jfnUXL7xzAssXmw7M7BsiqOkD3JLyfDJTi3C/mX0LfJHL9lp3PyT9/14z60tcQFszxt13S/+/y8wWAI42s/PcvaJaOHf/3sx2Ax4ETiYCxSmY2dLExbSfu5+eFo9Kx3OqmZ3l7i+a2cfADx3U1HGbuxc7M99jZusCG9F28AaAu79hZm+kl8+4+9jUYfUr4AB3fyStG21mvYj+M1lNwBbu/hXxGdgVWARYxd2/BO4ws7VpIdDNlWWimV0O7GNm+7n7d2Y2BxHY9ivneEREam26bOJJXioGFsAfgP8Cz6amhaLhwFFmNlN+Y3e/DLjMzGY2syWAXsDyKf+p0rdgFeAnRFNM1jCimaEPkwOU93P9Gor9WIq1EPcBp5jZg0TNwu3ufl6JPB/MvX6LqJ1pzdW51zcQTSJLELUhFXH3R8zsPOCvZnZdiWa04uiX/Hm5iqhVWRN4sdJ825AvwzhgmkbEpP5Ga5lZU2pK+hWwJLAqU39GXk7BSdGHwHcpOCn6FFiqzOwvIYLfTYhgdyuidjF/TkVE6lJXDFDKbeL5MPP/eYiL0Y+5xwBgRuDn+Y3NbJY0VPVL4FmiWaBn2q7cCqe50/MH2YWpVuJjotakKN8XovjeFY/3NKLPxqzERfwlM3vRzNbKbfdNif20dc7ey73+KD3/rI3tWnMU8DZwsZnNnFtXPC8f5pYXX89Jxyt1fqd5yLqZbU8c51tE36I/pbzyn5GvmFpZ/V9Kcff/AI8wuZlnJ+BWd/+svfsUEammQgWPRtGei8qXRKfTlVt4fFJim3OIWoStgNndfXF335EIUMr1eXqeL7swdXj9RQv5luTuze7+T3dfMe1vV6J25gYzm7GCMpUyT+71vOn5o3zCcqVOuXsQHYMH5VYXz8u8ueXzp+eyz0stmdlqRN+Z64Ffuvs87r4O8GiVinAJ0NfMlgRWJzpwi4g0hOm5D0rW/cAGwHvuPqm2wMyOJ/oB7Fxim9WIkSC3ZtKvCPRgyiCptY7IY4iRNNsxZRPDlkTNzUPlHkBq2nna3Q9094+AIakD7dlMbgZqr42IzqrZ8r3t7m+0kL4s7n5PqoU6BPg+s+r+9LwdcHpm+XbpuXhe6r2T9++Jz8IAd/8aYjI3ok9TNWY8vo4IpP9N1NKNqEKeIiIdolpf8Ga2HdCfGDE6FjjZ3S9vJf18xMCIvkSNvwOnuvv1beXVngDlUmIo591mdhLRH2Uj4GBgUOpIm9/mcWArM9szFe63xAEWmDIg+AL4fZpVdIr+H+7+mZmdRnQ4/RG4g5h4bBBxkb6rgmMYTQzf/ZCo2l+QuPDf4+5fmNlcFewr789m9gExzHUTYrRRRTPqtuJQIjhcsLjA3V8ysyuBE81sVqLGoTdwNHCFu7+ckn4BzGtmGwDPuvv7HVSmjvJ4ej7PzC4jPsj7EZ+VJjObpTiUvDO4+9dmNowIsE9z93oP6EREJmmuQuONmW1F9G88h7jmbkr0L/3W3YeVSP+TlG4uYsTle8SP9qFm9md3b7WfX8W/TNOv29WBx4hhvncA6wP7u/vAFjY7mBiefBLRmXZ3YsbYC4mApFiOAUQz0Z1kLsKZvI8hRrJskPZzMDGs948VjpAZmPL/C3HyzkzPW1ewj5b0B1YiRgv1AbZz96s6YL+kTqB7l1i1K3Fu/0Kclx2JY9w1k+YCYmjwLXRcwNRh3H00EZCsTrz/ZwLvEMOQScs7W/FWAEOqkJeISIep0lT3JxOjbg9y9xHuvg/RYnB8C+k3IH5kbuXul7n7KHffi7jeHt5WZk2FQiN1malfmUnMdnT3K2tcHGkHM7sEMHdfdVr3tfx8q9blH9YLn42tdRFKWqXHVLWudWGmpu61LkKLRj47uNZFKOkPv63ZxNetmqN7fnxB/Rj+zu3T3DPkuEW2L/s759i3r6o4PzNbDHgD2DrbPJNqVYYCi+VnBU/TPWwD7J0ZCYyZnQXs7u6zt5ZnXU9135WkicjK+babmH0jO6ks3Sij9szdK7pRYz0dYyXM7EDg18AuRGduEZGGUknNiJnl5/yairvnuzosWVyVW/56cbfEj/TsPu4F7s3lPSMx0Wubk2BWo/OhhJ2Zemh2qceaVSjLseWUJdUKVWLNcvZL6Y7UtbQG8GfgdHe/qdaFERGp1ISmQtmPdipOWZG/eW1x2oc5ytzPqcQ8Vye3lVA1KB3E3cfS+pwutxH9a9rcVYcUqHUXMLm/RWvyc7q05SnKO8ZOvzlgJdxdtSYi0tAqCTtK1I6Uo3h9y2dVXN5qJU6qYT+VuJfeae5+S1sZKkCpknTTuk9rXQ6ANDy80uCjnP1+BTzZ0fsVEZHWVWEm2eJM3fmaktlz66eSRvMMAbYlgpPDyslQAYqIiEiDq8Iw42Ltfi/ghczyXrn1U0j3NhtO3Lbkb+5+TrkZqg+KiIhIg+vsqe7d/XWieX7L3KotgNfc/Z38NmmyzVuIe9htW0lwAqpBERERaXhVulngccClZvY5USuyCTF/2LYAZtYDWJy4oet4Yt6uPsR8Ze+aWfZmuwV3f6y1zBSgiHSCpqb6vOHFsvMsWusiNJT/FSbyQ3Mltwyrnnqdb+Se5y6sdRFK2mD5fWpdhE41sQozybr7kNSf5FBiwtU3gZ3c/bqUZENitvm1iBnbiwMQ9kqPKYvcRgyiAEVEpAX1GpyI5FWpBgV3H0zUiJRaN4TMTNzuvva05KUARUREpMEVqlCDUm0KUERERBpctWpQqkkBioiISIOrxt2Mq00BioiISIPreuGJAhQREZGGN6ELhigKUERERBqcOsmKiIhI3VEnWREREak7XbEGRffiSdKtoEUm0WdCRBpFcwWPRqEaFMDMegP9iWl6MbOexE2RdnT3K6tcloHAgDaSve3uPTsov17Aa8B27n5tR+yz0aWpnE8BxgDXtZFcRKTmJha6Xg2KApSwG7B05vX7QG/g9RqU5SLgrszrY4Blga0yy36oaommP/MDfwN2rHVBRETKoXlQphPu/gPx67kWeY8DxhVfm9nHwA/uXpPyiIhI/euKfVDaFaCY2QzAX4E9gMWIGoeLgVPcfWJKsyNwMGDAR8BlwHGZ9b8DjgdWAb4nag0OdfePzGwX4o6IC6ULdjHfscDd7r57phlmG2BXYE3gA+Acdz8ns00P4hbRfyR+GX8N3Acc7O5vm9kQYOeUtpD2NZpcE4+ZLQWcDPwemBV4EDjc3Z9P6/uk/a4NHJ3SjSdunHR08bg7Wsr3eGCldGw3Aoe5+5eZNEsRTRZrEPP53A8c4u5vZna1kJndCPQlamiuI96Pbysoy7zA2cQ5mAN4FTgjcw5PSPucObPNDMCPwJHufoqZrQOMAtYBTgCWI+6YOcjdh6Ztis1SWxGfwdWJ9/5Mdz8vs+9ZgaOIz8hCwBvAWe5+USbNOOAa4P+A5YG7gc3S6ivMbKC79yr3HIiI1EIj9S0pV3s7yV4MnAoMBTYBLif6TVwAYGb7pWWPAZsCZwKHExd4zGx54iLZnahG/ytx8RzejrIMBt4DNk/bn21mB6R8moA7iQvm4cTFdyCwLnB+2v544FbiAtcbuD2fgZktAzxBBDh7AzsBPwceNrNf55JfQwQ4GwJXp3x3acdxtcnM1iIu5p8DW6a8/gTcYWbdU5qFgUeBhYE9U1l6AaPMbJbM7k4iLuCbAP8A9iEu7pW4Glgi5bMh8AJxkV+tHYd3PXEeNwNeBK41sw1zaS4E3iHe+zuAc81sXwAz6waMID5b5xLHdS9woZkdmdvP34gas62Iz/XmaflA4ryKiNS1ZgplPxpFxTUoZrY0cYHu5+6np8WjzOxb4FQzOwc4Frje3fdO60ea2c+AdVPQcDRRq7JBak7BzD4DBqdfx5UY4+67pf/fZWYLAEeb2XnAAsBXwAHu/khKMzrlsRuAu7+Rb0Yxs9lyeRwLfAOs7e7fpDQjiQv6IKbsHzLY3U9I/7/PzDYFNiKCuo52CvA8sKm7N6dyPQc8SVxYryNqsboD67j7pynN68QFfXnifQC4yt37pf/fa2brE4FdJdYgaotuSfncn/Y/oR3HdpW7H5n2M4IIfPozZQD5kLvvkf5/l5n9EuhvZucDGwOrAZu5+80pzcjUAfYYMzvf3b9Iy98CjnD3QsqvZ1r+hrs/246yi4hUlZp4whrp+Zrc8quIX5+rA78AbsqudPdBxMWc9Iv6lmJwktbfDSyeWV+uq3OvbwC2AJZw91eBtcysKV10fgUsCawKzFRBHmsAtxaDk1Ter83sVqKGKOvh3OtxQD7gmWZmNjuwMlED1C3VGAA8B7xL1BJdR1ykHygGJ6nsLwM9036KAeGDuSzeIppXKjEaONHMViKa7O5w90Mr3EfRpPfV3Qup+WlgCjCmSpPcQNQgLU68Z98Bt+TSXEU0C61C1LAAvFQMTkREGlFXHMXTniaeudPzh7nlxddzpeePaNk8bayvxHu518X9/gzAzLYH3iYuuNcSF7BvgUrmuJibaALK+xCYM7cs32ejmc6Zb2Zu4hiOJfpwZB8LEbVHUP65/ib3uj3l3orog7IKUWP0XzO7w8wWqnA/UPp97Ub0bWktDcR7PzfwYYnAo/g5nbPEMhGRhqQmnvB5ep4X+G9m+fzp+av03CO7UepA+RuihuHLEuu7ARsQfT2KZ7B7Lu+flijPPLnX86bnj1JNzOXERfNMd/9vyuvvRH+Tcn0OzFdi+fzAJxXspyMVO8GeQtQc5I3PpOuRX2lmfYGXO7JAqcmkH9DPzJYkapeOIfq0bEa8r+W8pxDv69uZ1/MSTUWfMzm4aPG9T+nmNbOmXJBS/JzW6n0TEelw6iQb7k/P2+WWF1+PBj4l+gBk7cnk6vaHgPXNbMbM+lWJTq7G5IvrpF/eZmZMfUGC6N+RtSUxkdkbxEiabsCATHDSnWj+yB57WyNs7gc2zvZNSf/fOB1L1aVg4HngV+7+ZPFBzN1yIjEqhVS+1c2sWLOFmS1GNG+s3lHlMbNFzGxc6nODu7/q7qcQHVOL7+N4YAYzywZ7LTXnTXpfU7+lLYimqgml0iRbEv1G3ibes1mIGrOs7YhRSk+0cjidMuJKRKSzFCr41ygqrkFx95fM7Eqir8GsxAiR3kTH1yvc/XkzGwScY2afALcRNSdHAqe5+/dmdjzwCHBb6sz6U2IUyWiihmV2oqnkbDPrT/xiHgR8VqJIfzazD4jhoZsQv9R3SOseT8/nmdllRLX/fsBvgSYzm8XdvwO+IH5tbwCU6hR5HDEi6R4zO5VoWjkslfu4ys5ghzoauMXMLiFGVM1MjLxZkhiZAnA6cT7uSmUvEOfyJSJgXCC/0/ZIQ7bfJc71XEST2srAekQzFEQH178Dl5jZGUQ/mGOJ4dF5R5jZD0Sfmt2AXwN9cml2MrOPiCBoMyJg3DatG058loaY2bHAK0RAszsw0N2/omXF2ql1zOw/7v54K2lFRGqukZpuytXevhG7EgHFX4gLwY7EkMxdAdz9XOJCsC5xUdqPuBANSuufIkaIzEIMJz0LGAls7u7NaQ6PLYgA6hYiCDiO0r96+xNzgNxCXMC2c/erUj6jU96rE8ONz2TysFSYXINwATHXxi1MDm4mcfcXUtrxwBXEHC2fAL3d/cWyzlgncPfhxHBeIzolX5LK1cfdX0lpxhK1FJ8RZb+QCE76VjLHSZk2IwLFk4j3cy+iiefUVJaXic9IL2IU0X7EZ6hUH5C/EfOX3EQMkV7P3fMdkI8CfkcME18N2Nrdr0t5TSSaDC8nguPbgLWAPd291aDS3ccTQ+K3BG7PdEAWEalLhUKh7EejaGqkwmbV8n450nkyE7X1bmn23Ea4f9AK86/WmH9YNTJLt0oG1VXPD80/1roILZq1Ts/ZPc9dWOsilLTB8vvUuggtuvvdEdN8Y9K+C61f9nfOyHfvaogboWqq+ypJ/SjyHURLmVhPQ14btdwiItMTNfHItNiZqYcDl3qsWasCtuAPlFfu7WtVQBGR6V1XbOJp2BqU1LeiIaqpktuITqNt8c4uSIUeo7xyv9l2kralCftafV/d/fW20oiITE+qVYNiZtsRfT8XA8YCJ7v75a2k/ynRD3ELYmDJA8CB7v5aW3k1bIDSaNJMrp+2mbDOpNEuT9a6HCIi0rJqDB82s62I2bjPIWYL3xS4zMy+dfdhLWx2HfEjtx8xT9oA4jYwS2dvaluKAhQREZEGV6Wp7k8Ghrr7Qen1CDObm7jlylQBSpos9Y/EfffuSsseJAa47E0a4dkS9UERERFpcJ091X2a4HNxpp65fBiwpJktWmKzvkStyajiAnf/mJhI849t5akaFBERkQZXSeBhZl+0lcbd58otWrK4Krf89eJuiZqR/Davp3mp8tts01YZFKCIdIJ6nddjxqb6rDRtqtM+zzN1787LX71b62KUtPKci9e6CCXV63wjdz5zfq2L0KmqMDqneA+08bnlxVm552Bqc5ZIX9ymVPopKEAREWlBvQYnInmV1KCUqB0pR/FXRD6j4vJS9ytsKpG+uLzN+xvW588pERERKVsVbhZYHHGTr/mYPbc+v02pmpLZW0g/BQUoIiIiDW5iobnsRzsV+570yi3vlVuf32axNCN5fps25/xSgCIiItLgOnsm2TRB5lvETVSztgBec/d3Smw2EpgLWKe4wMx6AGsQN5ZtlfqgiIiINLgqzSR7HHCpmX0ODAc2AbYGtoVJwcfiwMvuPt7dHzCz0cC1ZnYY8BkwEPgCaLPXsmpQREREGlwV+qDg7kOICdbWA24G+gA7uft1KcmGwKPACpnNNgduBU4HhgDjgD+4++dt5dfUSDcOEmkUqy64dl3+YWmYcWXqeRRPvQ4z/r4wodZFKKmehxnP+PPFpvkP4Dfz/q7s75wXPxxTn39wOWriERERaXDVuBdPtSlAERERaXDTMDqnbtVnfe90oMSwKxERkXZpLhTKfjSK6bYGxcx6EkOmdnT3K6ucd2+gP9GhqKZlSfkPAXbOLf4KeAk4IeF1qwAAIABJREFUzd1v7OD8+gD3Aau7+0PTuK+xwN3uvvu0l0xEpDF1xSYe1aDUxm7A0pnX7wO9gbtqUxwgelb3To9Vge2A14BhZrZuDcslIiJtUA2KdAp3/wEYU+Ni/ODu+TLcbma/B/Ykc7tsERGpL12xBqXLBChmtgdwEDFJzHvAYOBUdy+k9ZsDA4AlgFeICWey2w8E+rv7DLnlBeAYdz8hvZ4fOBXYAPgJ8ATQz92fTut7pH3/EZgf+JpozjjY3d/ONqekfe8KjCbXxGNmSwEnA78HZgUeBA539+fT+j5pv2sDR6d044lx5keXuL11e31B5mZPZtYdOAzYnjjXzcAzxLkbnUn3O+B4YBXge6J26FB3/yiz71//f3v3GSZZVbV9/D9EEQQBhQEEAcUbMTyi4kMakgIiEgVEVKLI8wqCIgIz5DCEARSRICDBACJBchqy5KwCwi0ZyYLkLNPvh3V6pqamegJ2nV1dtX7X1VdXnVPdZ031zJzVe6+9tqS9gOWJfRlOJN7rd6vvMR2wKzHi9BHiPTrE9gkDBStpLmAf4OvE+38PMLpxmqp63/cE1q3+DPsCh1efvw3MT/wd+gOwl+13pvrdSimlAt7tG6z/8jtHV0zxSBpJJCSXAGsBvyZuNodW59cCzgT+RtyUTgemudZD0mzA9USb3p8AGxLv4eWSPlIVvl5MJA27AKsRXfNWZULXvP2IpjVPE9MpF7a4zmeIxGc+oinOpsCHgOslLdH08j8QCc6awKnVdTef1j9bdd0Zqo8ZJc0taTvgc0zc8e8QIiE6BvgqsHUV2xmS3l99nyWBa4Dpge8C2xHv2QVNl/xFQ+xnACOr79fvGCKpPJn4uV4AHC/phwPE/37gOqIV82hgPSIZPUvSpk0v35N4774LXES8bz8gkpvVqmvvDIwa6P1KKaVO0e5W9yUM+REUSXMQBadH2d6xOjxW0qvAYZJ+QdyMbrb93er8pdVv0QdN4+U2BxYGPmv77ur6NwF3EHUb1xPFpdvbvqH6mqslfZwYBcD2g5L+RcOUiqRZm66zJ/AasIrt16rXjAUeJG6gGza89tj+0R3gKknrEqMHA44yDOBjQKuRgiOJJKLf/MBI20f1H5D0JnAWUVdzK5HAPAusUU1fIenfwLHVe9HvMNujq/NXAesQyd2vJH2CSFZ+avuw6vVjqxGc/SSdYPv1pli3AD4JfMn2rdWxi6tRlTGSTmkYWbre9qENf4YVgduqTokA10h6nRhBSimljlZTq/taDfkEhRiFeD9wnqTGP8/5xLD9msAXmPQ34dOZ9gRleWJTpLv7D9h+ibi591tZ0rBqZc5iwOJE8jLTNFxnBeC8/uSkus6rks4jRoAaXd/0/HGgOeGZGo8TIw4Aw4gNnlYHdiQSlx2rOBr3XBDxZ1yr+rr+P+PywLn9yUn1dZdTvU+SPlIdvrbhfF+1IueD1aFVqjjOb/q5ngf8CPgSEydOEO/bAw3JSb9TiCm5xYkpH4C7ml5zFXCQpGura1xo+0hSSmkIGEojI1OrGxKUuavPYwc4Pydxo/tX0/Gn3uO1np3cCyR9m6gdWZDYGOlO4PUqhqk1FzEF1OwZYI6mY82jCON4b1N3b9m+renYZdUI1faSxth+WtIXgaOBpapr3wP072LZ/2ec4vtUea3peWPs/T/Xgbbknr/Fscm9bzDxe/dM02sOIeqFtiRqjMZIugf4oe2rBoghpZQ6wlBanTO1uiFBean6vDExBdLsSaLuY96m43M3Pe+j6cZe1Zw0X2vB5gtIGkHcGOcFfkuM3PzM9hPV+THESM/UegEY3uL4fMBz0/B9BsOdwPeAhaspj0uAvxDTOffZHifpa8SW2/1eAj7c+E2qgtc1iCmgqdH/c12RSZMwiILZZi8AS7Y4Pl/1ecD3zvY44CjgKEnzEEXOuxH1K/NmoWxKqZN14yqebiiSvQl4G5jP9m39H8CMwAHEzekGYIOm7q1rNX2fl4FhDdMPEFMVja4DFpO0eP+BKok5n0iQliXe070akpPpiSLZxvd6SuXW1wBrNdamVI/XqmKo05eIeB8kpkjmBn5u++/VTR0i8YAJf8brgK9KmrHh+yxHFLlqKq/75+rzXE0/14WIAuhW01jXAB+XtFTT8W8RCeQDA11M0rVVvRK2n61qUY4kRuDey5RZSinV5t2+cVP9MVQM+REU289JOgw4oJqOuBb4KLGK4yViCmIUcCXRdOx44kbbXJNyIfAz4IRqxGMhYgXJKw2vORHYnqiL2IuYwvkpsYz210y4+R4p6TfElMO2wP8Qyc8stt8gCi/nlbQGMRrRbF/gZuAKSQcTUyc7A7PRtDx6EM1cLQ3uNyNRv7Mp8Gvb/5L0NpHI7VEVGb9LjJxsVX1N/418PyIpPF/SkVXcBxA1I/2roCbL9t8k/QE4UdKixEjOp6rvc7vtx1p82cnAD4l6pD2IuppNiATqew0JVStXA7tKeqaKfQFipdYVtrNQNqXU0bqxBqUbRlAghuJ3IUYxLiaSk0uAlW2/afta4ia1IHA20Xhsy8ZvYPsfxM14YWLZ6Q7EKpInG17zMnFzvZ2YDjiNSE5Wtv1U1QdkW2BEFcfPiPqM9atvMaL6fBzwEHAu8J3mP4ztu6rXvgz8DjiJmJ5YprFAd5B9BLix4eMyYG0iSdu2iuslYqXN9MSy7d8SidwKRCI3onrd7USR6yzE8uGfEzVC608hSWi2GXAEsUz5UiJJO6GKaxJVUfGKxM/+IOJnvTjwjcn1TqnsDexP/L24hPjZXQJsNA3xppRSEd3YSXZYN2ZdKZW23AKrdOQ/rBmHdebvJMOmqYa8Pn9/5Z+lQxjQUnN8bMovKuDNvv+UDqGli+88ZsovKmTGDy36X/8DmHO2j0/1/zkvvPpAZ/6DazLkp3hSa1W9zfRT8dJ3+7vtppRSGpq6sQ9KZ/46lQbDZkT/kil9rFgqwJRSSoMjO8mmoeR8olfJlAzUZySllNIQMZRW50ytTFC6lO3ngedLx5FSSqn9hlLx69TKBCWllFIa4obS1M3UygQlpZRSGuK6sZNsJigppZTSEJcjKCmllFLqON1Yg5KN2lJKKaXUcbIPSkoppZQ6TiYoKaWUUuo4maCklFJKqeNkgpJSSimljpMJSkoppZQ6TiYoKaWUUuo4maCklFJKqeNkgpJSSimljpMJSkoppZQ6TiYoKaWUUuo4maCklFJKqeNkgpJSSimljpMJSkoppZQ6TiYoKaWUUuo4maCklKaKpI9M4fwadcUylEiateHxepK2l7RoyZg6laR/SPrsAOeWkvRM3TENRNL0kmYvHUc3m6F0ACmliUn6ALAdsCowH7ABsAZwh+0rC4b2N0k/sH1a48HqP+lfAJsC0xeJbEIs7wO+BMwPXArMavvxQrEIuAA4DdhD0n7AKGAYcKCkVW3fUCi2T9q+t8S1m0naEJixevpxYG1Jn27x0q8As9QWWANJMwAjgQdtnyppJeBMYE5JlwPftP1iidi6WY6gpNRBJC0A3AnsXh36BDAzsAxwsaRVSsUGXA6cKulUSR+E8aMm9wAbETffYiRtCzwJXA2cAiwCHCvp8sZRjBodBPwHOFfSTMC2wOnAB4nkaXSBmPrdI+lmSf/X/7MsaBng99VHH7Bvw/PGj82BI8qEyD7AnsTPDuCXwL+BHwOLAwcWiqurZYKSUmf5GfA2cXNdjfhtG2BD4Apgr0JxYXsjYBNiZOdvkk4jRgjuBj5t++BSsUnakrh5nQx8mQnv2wnAUsQNpm4rAqNs3wasBMwBHGv7ZeBXwBcLxNTva8CDwGHAk5JOk7S6pGFT+Lp22JX4+74o8XNbr3re+LEQMIft3Qf6Jm32LWCk7aMlfRL4FLC/7SOIxHzdQnF1tZziSamzrA5sbftZSeOnS2yPk/RL4A/lQgPbp0l6DTibGDW5E9jI9isl4wJ+Chxme+em9+1PkuYHdqo+6jQj8Vs2xBTda8B11fPpidGVImxfAlxSTSduBHwXuAh4StLvgN/Yvq+mWN4GHgWQdDbwpO1H67j2NJgfuLl6vCYwjni/AB4nks80yHIEJaXOMj3w5gDnZmDCyEDtJM0h6VdEcnI7Mby9MPB3SaV/g1wEGDvAubuB4TXG0njd9SUNJ0bAxtr+j6QZiRqjuwrENBHbr9g+wfZKwBLA/cAuxBTQtZLWrjmkrzBhGqWTPEn8XQdYG7jT9nPV82WJJCUNskxQUuos1wIjJTUWA/ZVn78PXF9/SOMZ2Iyoj1nG9i+ATwN/Ac6S9KeCsT1OFMe2siRlbiB7At8DngDmImpSAP4BrAzsXSCmiUiaWdKGks4D/kpMO/2GGFW5H/iTpDrrK+4k3ptOcyrwc0mXAMsDJwJIOpz4Of6uXGjdK6d4UuosuxDTAPcDVxLJyY8kLUH8hjuiYGyPAZvb/nv/AdtPAWtJ2gz4ebHI4oaxh6TXiboYgFmqEYDdKFBcafuyajXKl4CbGqYtDgOubHwf6yZpRWJa5xvE9MQNwA+AP9p+rXrZWZLeqY6PrCm0W4CfSFqfKL5uXlbcZ3vbmmJptAcxRbcCsKvtY6rjnwcOBvYvEFPXG9bX1zflV6WUaiPpE0Qx7CrA3MBLwDXAvrb/VjCu6WyPm8z5+aqEpXZVceeviBELiKmw/v/cTgM2tf1uzTFdBxxg+6IpvrhmksYBTwG/BU60ff8Ar9seWMH2BjXF9c8pvKTP9kJ1xNJI0o+Ai2z/o+5r97JMUFJKU03SdMA3mdCjZXtgaeD2kiMC/SQtxsSJ3Z9tF6n1kPQisK7tq0tcf3IkrQlcPLmEM00g6QXgO7YvLB1LL8kpnpQ6iKQVJnN6HPAq8FC1VLVWkuYALiGmLB4FPgp8APg2cJSkFW3fWXdcVWx7Ar+uRgLubzr3UeAntrevOaw/ArtIetD2lEYG2q5azdTvTmB49JJrzfaTbQ9qAFXzv/8lpp/+BdzWMPVUwoOAgExQapQJSkqd5WomTE00rthpHOocJ+m3wPdrnrY4hOhHsSTwd6JfC1QrVIh5+DVrjKfRXsDFxGqLZssQBcZ1JygLEwWfj0h6ldb1FANnCIPvcSb+ezQlRboCS9qFqPl4f8PhNyWNtl2qud05RPffNYnVWa1+ltmsbZBlgpJSZ1mHqJk4mfgN/GlgHqJ51bZEEe07wH7AI9XnuqwH7GT7b029Rl6RdBDRFK02VY3HMtXTYcBNkxkRuLGWoCb2JNHRtlNsybQlKLWTtBXRlfV44r17hphK3ATYR9KTtk8qENq+1eeVab3KqI/sJjvoMkFJqbOMBI6w3bhq4h/AdZJeAb5hewVJfcCO1JugvB94doBzbwLvqzEWiILYbxDJyb7AcUy6nPhd4EXgjHpDA9tb1H3NybF9cukYpsKPgV/a3qHhmIGrJb0B7ADUnqDYzpYcBWSCklJn+RwDt7O/jhhBgRhmnuzuwm1wG/D/iKmUZhsDd9QZTNXpdDTEzrLA8SXrJgZS7QK9CtGN9GRiROCeqoNqnXGMAk6y/VT1eHJKTVl8DPjRAOfOB7auMZZJVKvFFqeqjbH9YMl4ul0mKCl1lseArwOXtTj3daLpF0Rn1OfrCqqyB3CZpNuJYsE+YCNJuxPdNVevM5iq6PNZ2/8hpgSaC0EnUiJ5kXQI8Vv/DMT7NZaYClhA0iq2BxqRaof9iQ0fn2LKfTtKTVn8kyhGvbzFuU8CL9QbzgSSvgOMAeZtOPY0sNsQGZ0acjJBSamzHAocJ2le4E/ECoZ5iNqUjYFtJX2MmNq5tM7AbP9Z0qrEjWsUMbXyU2Lk5Ou2r6gzHuJmtgzR3GtqCkBrLfqsij23J/YAugB4oDq1N3AmMfpT24hA4zRFB09ZnA7sJ+kx2+f3H6wa7u1D9G2pnaT1qmtfzKS1MSdIesH2uSVi62aZoKTUQWz/WtK7xDTPRg2nHgY2s/07SRtXz3dp9T3a7A5i1c4LwIeBzYm9U14tEMuWxPLP/sedVgC6DbC37SOaiopvrEad6qwfGipGE7tAn1t1BX6GGLGYheh2u1uhuHYDfm9706bjv69W1I0EMkEZZJmgpNRhbJ8k6Xyitf04YrXOMOD9kr5n+9fESp9aSfpf4jfIY22PlDSSWL77EvBDSes3/tbbbrZ/0/D45LquOw3mB24d4NwjRDO5IiTNSYzkLEPrzfnqXgINgO03ql5AaxPbOsxFJMPXABcUbCz3KQZOjk4hRjvTIMsEJaUOIukzxH94nxrgJX3Ar+uLaCL7A/cRU1DvBzYFjrG9naRjiU0Ea0tQmlU33f8lbrjNUxh9tv9Qc0gPEnU5reopRgAP1RvORI4npg0vJva86Ri2+4gRlLuAOYk6o9KN7p4iEs5WPkLs05MGWSYoKXWWQ4jfrHciimLfIm76X6s+VioWWdz8v2n7YUnrEsuK+3dxPQ34TqnAqtqYPxFLoYe1eEkfUHeCcjjwK0kzEj/DPmBRScsDO1Nmiq7fV4ieNr8oGENLkv6PSHbnazj2IDDK9pmFwroQ2F/SX22PX60m6QvEEvdiiXk3ywQlpc6yDPBj2ydKeg34drVz6jGSziSKLq8rFNs4ot8JxMjAi0SBKsDswOslgqqMIUYsdiTqc4rvMWP7eEkfIqYGfkgkTqcTHXgPs31UwfBeJfqLdBRJ2wK/JOo5ziL67sxL1D39UdKGtktMp+wJfBm4VdIDRAPF4cDHiT5FuxaIqetlgpJSZ5mZCXvJ/AP4n4ZzJxE79pZyG7B11TBrI6ImoE/SPMR/0LcVjG1xYH3bVxaMYRK2D5R0FJF49m9geJPtupeINzsS2EnSdbZLFDgPZEfgyBb7Jv1O0jFE8XjtCYrtF6rRki2ZUBvzF+AXwMm2SybnXSsTlJQ6y2PAIsC1RIIyu6SP2n6UGL2Yq2BsOxObBW5MLH/u76VxNzE6sFqhuCCWHL9/iq+qkaQTgf1sP0zTknBFT/4xttcpElwkKJsBj0u6j0lHv/psf7n+sBhOLMlu5U9EzEXYfgM4qvpINcgEJaXOcjZwkKRXbJ9d3Tz2k3Qg0Qa8WOdK23dUPViWAO5u2F32+8D1tv9VKjbgIOJ9utN2seJTSQs1PN0MOKdaNt7sa5RN6I4nGqLdDbxSMI5m1xLbF4xtcW4V4Ka6ApF03DS8vM/2Nm0LpkdlgpJSZ9kHWIxo4HU2kZScDXyb2Fdm43KhxcaAwM1Nx84pEYuk+5m498miwP1Vd8/mVRV1LZs9ikg++p09wOuGUbaw8uvAjrYPLxhDK8czoVHhH4gNF+cm4t0UGCVpfH8g26e3MZbVmPreOp3Wg6crZIKSUgep5rLXlzRz9fxSSZ8GvgDckXt/TOR6Jr4xXF8qkAbbELvdDiM6j+7NpKNe/RsYXlVrZBN7lRg96TT9mzquXX00G9PwuI8oOm4L2wu363unqTOsry8Tv5RSGmySNiMKiUsXxE5C0l5E4e76nVTgWU0hTrW6E3ZJ7wO+RPREuRSY1XbzDtppkGSCklIakiQ9RNQqXAZcZfvfhUOahKRFgPfZvlfSHETPjAWBMwo0jmuM6ygm9K35O5PWofTZrnXzx2ZV/5jZgRcKdpBtjGdbYnuCDxKjN0tVz2cG1mmoyUqDpFM3jEoppSn5K7A+MS3wrKRbJI2WtFJ1cytK0hpEr5GtqkPHAj8AFib2cNmyUGgQhc53VB9vAjM2fcxUKjBJa0i6nlhZ9CzwhqSxkpYuGNOWwBHAyUQ/lP5mgCcQico+ZSLrbjmCklIa0iQtDqwALE/0qPgocXP7MzG6cpnt2ustJN0APE/seDs9sfHdwbb3lLQ/sLbtz9YdVyeTtCHRlfivxLLiZ4mlx+sDnwRWs31NgbjuBc63vXO18eM7wBerlW3bEV15F647rm6XRbIppSHN9n1UewQBSPoIkaiMIFZ+jJH0L9sD7aXSLv9DJCGvSPoW8f9tf6v2y4Cf1BzPVKkKtEfYbrWHULvtBZxu+1tNx/eVdAaxnHyZ+sNiEVovfYYoNh5eYyw9IxOUlFK3+SDwIWJ56izEcPwbBeJ4gxg5gdga4Bnbf6ueDydW8hRR9Ws5GliRmM7pn7KYruHx9C2+tN0WJbrJtvJryu0a/DhRHNsqaVuyOp8GWSYoKaUhrdrvZjUiCVgNmIco+ryKqBsYa/uBAqFdD/xU0lzABkT9Qv8Gc3sRTclK+RkxwnQisBwxJXYj8f59hphSKeEvxHRdq9GK/yFGyko4EdhD0utM6HQ7i6S1ib2WjigUV1fLBCWlNCRJGk0kJZ8jVlXcShSiXkbsd9Oqg2udfkzsgnsqsVKmf2uAC4k+JCU3mFsZ2M32kVUNxdq2d5E0ikgO1gHOKxDX7sDpkmZj0kZtuwI/kvSl/hfbvqXldxl8BxK1TYdVHxA1ThA1M6NriqOnZIKSUhqqRgLPAaOAX9l+uXA8E7H9kKQlgHlsP9Nwai3gL7bfKRQawGxA/3TTfcSIDrbflXQ0cGihuPqnULYndoDu1z/tdGzD8z5qmoay3QdsI+kwouX+XMTGj3+2fVcdMfSiTFBSSkPVqcSSz4OAbSWNJZpnXWH7haKRVaob2zNNx26VNLOkrxQqRAV4Cpi3enw/MJek4bafJlYezTvgV7bXqoWuO1Vs/4PYxDPVIJcZp5SGNEmfY0L9yXLEb9V3EFMVY4EbSkz3TE0hqu0ShahIOgZYCdjc9s2SHgX+SLTmPwpY3vZiJWLrRJKGEftjrUIUYTf3ECve2K4b5QhKSmlIs/0XorjyYEnvJ+orViUKPUcBr0q6yva6NYfWqYWoAHsAlwAHEKNQo4DfMGHp87YlgpK085ReY3vMlF7TBvsTU4r/BB4Bine27QU5gpJS6jrVni5LA18FNgRmrHu0QtLzwF5NhairVY2+xgKP2N5q8t9lUONZHrjd9hsNxxaw/UTD+WWAW0o0Q6timNyN/xViqfYn6oqnn6SniP4sO9R97V6WIygppSFN0kzAF4FliZGKZYAPE31G/gzsBFxRILROK0S9AFgTuF7SlcAPqiZ3VHFdB1xXc0zNWm1RMBsxEnUksVt0CXMAZxe6ds/KBCWlNCRJGkMkJJ8najzeInqPHE4kJLcX3mSu0wpRhwFfkfQ4UX/yiaqvR0u2H6srsIZrtqoVegm4QNK8wCFEMlq364gRuasLXLtnZYKSUhqqfgzcRvSluAK43vbbZUOayMVEi/bHqkLUx4EdJe0NbAY8UXM8ZxKjOHsSS3SnNCJQpIB3Mh4GPlXXxSQt2/D0DOBwSbMQycokOxfbvqGu2HpFJigppaFqLtuvTMsXSJqO6LWxje372xPWeJ1WiLo1cDqxDcDviBU7D9Ycw3siaR7ifXu0xsteRyRy/YYRP1NaHK+tJ0svyQQlpTQkTWtyUhlGTG98YHCjmZTt54AvSlqgen5KtZy3SCFqNd11KYCkrwC/s/3wlL5O0qbETr5t7y0j6Q0mvvlD3PhnIH52m7c7hgYr13it1EKu4kkp9YxqBc07wBdt31E6nk5XvV9vA0vV8X5J2p9JE5Q+4GXgQtv3tjuGyZG0qO2HqscfAmT7+pIxdbMcQUkppUEi6cRpeHlfncuM/wvDpvySwWF797quNS0kfRg4n9gXqL+B3ZeI4t0rgW/YfqlUfN0qE5SUUho8mxO/8T8G/GcKr83h6xaq4tT/2L6lmh47AlgQOMP2IYXCOpRYdfW9hmMXE12CTybqjIo0t+tmmaCklNLgOZXYDHB24Cxip9urqj150hRI+jbwW2Jl1i3AcUQScCWwv6Q+2yU2MvwqsK3t8f10qp/ptZJ2q+LNBGWQNe8nkFJK6T2y/R1gHmArohD3XOBJSUc0LVtNrf2EWGG0S9X3ZHVgX9trA7sRK5FKeB/w5gDnXiH250mDLBOUlFIaRLbfsn2O7W8Rycr2wHzAZZIek3SopKXKRtmxFgdOrkYnvkbUv5xTnbsFWKhQXDcDO0iaaNahKiLejogtDbKc4kkppTap9r05AzhD0qzE9M8GxNTA47Y/XjTAzvMy0doeYlrlMdv/qJ4vQnTgLWFPoovsg5IuAp4ltlP4KpF8rlIorq6WCUpKqZf0Ec2+3ipw7Y8BS1QfM5H//7ZyFbCXpMWB9YCfA0haB9iP2GSxdrZvkrQMMc20LrGa5yWimdsGuWS9PfIfSEqpa1Q3kVWB+YmVFZ8E7rT9LIxvVrZIjfEsSYyYbEgkKE8QLee3tH1TXXEMITsAfyB+dlcDo6vjRwL/BHYtExbYvpP4WaaaZKO2lNKQV+1ofAqwPtGIbUZgKWAMMWIxwnYtbd0lfZG4kW0ALMqEpOR02zfWEcNgkrQicJvtSfafqTGGhZo3L5S0CXCR7RdriuF9xF5AMzGhN8x0wKzE36+O7OEylGWCklIa8qqdjbcBvgNcBrxO7Hr7b6JfxV9tb1xDHA8BHyWSkrPo0KSkutmOBL5O3GCbF0z02VbtgU2lAh1uV2TCPkatvGI7V/IMspziSSl1g28DI22fX928ALD9SLV78C9qimNhos7lbWBNYE1pwPt8ySTgF0TTsauBu4FxheL4b9TW4RbYn0h2/49Igt8FTiJWGv0/YI0aY+kZmaCklLrBXMADA5x7nmicVoff1HSd/9YGwCjbB5cOZIhYEvie7bMlzQH8n+2LgYur6cXdiYQ0DaJMUFJK3eAeYGNar/JYA/h7HUHY3uK9fJ2kFYDba6zzmIns3TEtpiOm7QDuJ2pR+p3F0ElMh5Rs1JZS6gajgc0lncOE/XCWk/Qz4IdEsWxHqqakrgLqnO4ZS05LTIsHmZCUGJhVE+bupie6BqdBliMoKaUhrxp6/w5wELB2dfgXwL+IPVROLxbc1KmzngLg98Dxkj4E3EAUFU/E9qk1x9TJTgXGSJrO9tGSbgOOkHQ4sAcxgpcGWSYoKaWuUN1QT61+s+1vpHVv1fskTeys6vPm1UcSbLWVAAAceUlEQVSzPuKmnMLBROfY5YGjgR8Qq8MuJLrfrj3wl6b3KhOUlFJXkPQ1YGXbP62efwm4VNIBtq8qG13Hqa1ZXTeoktwdG57fJmlRYu8g2365WHBdLBOUlNKQJ2kjogPpJQ2HXyPq7MZKWsv2JS2/uAfZfrR0DEOZpHmIJeUPZnLSPpmgpJS6wW7AUba37z9g+x7gy5J+CezLxMlLz5F0HHBA1RvmuCm8vM/2NnXE9V7YflfSqkTBattI+jLwfWLK62jbf5Z0KLFD9fTAu5JOALaz/W47Y+lFmaCklLrBx4EfDXDubOA9Lf/tMqsCR1WPVyNuugMp0mK86nA7itgd+IO07nD7KQDbV7Q5lvWILQoeAl4ELpd0DLAdcAJwJ/C/RALzT2L/oDSIMkFJKXWDZ4AvEMt1m32W6ALa02wv0vB44an9OkkLAU/a/k874mpyGNGZ9Sbgr5TtcPtT4He2NweQtB2xMuww2ztXrzlG0tPAJmSCMugyQUkpdYNTgL0kvQKcAzxLrLpYC9iHWHmRplHVo+VhYuPFtu95A3wT2Mv2fjVca0o+BTTGcSpwBHBp0+suIKZ80iDLRm0ppW7QX2NyDPAk8B/gKeB4YvPAPcuFNnlV7cIWRCLQiers0TIzcG2N15ucDwAvNDzvL4Zt3j35HSLuNMhyBCWlNOTZfgfYUNKniV4VcxF9UK6z/dcSMUkaRvQYmdyOwasD2M5W6aG/w+3VhePo1zjF1Nf0ObVZJigppa5h+25id96JSJrN9qs1h3MAsAsxMvI4Q3PH4Larloj3uxEYXXW4vY5YKj6RmrsCt0pGMkGpSSYoKaUhr9pRdntgRWIjvP5piemI0YvPVp/rtDnwM9s71Xzdoea0Fse2oPXKqz6gzgTlTElvNR07p+lYTu+0SSYoKaVucDCwA3AXMA/wBrEPz2eIhGXvAjHNDpxf4LpDzWKlAxhAq2m362uPoodlgpJS6gYbEMs/fyppFPA52xtJWgC4hjILAm4AlquunwZg+8HG55JmBpa0fVP1fH5gBHCu7TdrjOs99c6RtAJwu+1JpqfStMkEJaXUDeYlNm+DGEX5PoDtJyQdCPyEWOlTp9HE5oUzMPCOwTfUHFNHk7QIsepqOmDR6vCniG0M7pK0uu2nS8U3JdWy7Kuob1l2V8tlximlbvAiMZUD8ACwoKQPVM/vBxYqENOVwHBiemkssXy2/+M6Omc5bSc5jEjk1uw/YPsyolNwH3BIobimRZ3LsrtajqCklLrB9cAPJV1DJCSvAesCvyPakb9UIKaVC1xzqkjaFLjQ9vMtzg0Hvm37MGLl0T5Eb5k6rAhsZvvexoO2H5K0N3BsTXGkDpAJSkqpG+wN/Bm4yPbKko4Gjqvak3+eaOBWK9udXHtyErA0MEmCAnyOmJ46zHYfkaDUZRgTRsJamaWuQFJ5maCklLrBd4m29v03sJFE58/lgP2BA0sEJWlx4ga/EjAH8BwxtbNv8yhBDbFcACxRPR3GpMtl+81LjEKVcC2wm6QrbY/v2CppdqKnzJ8LxZUKyAQlpdQNtgYutX0pQPWbf9HN2yR9hph6eh04l9jQcD4ikVpL0jK276oxpP2BrarHWwG3EkuxG71L1PMcV2NcjXYhmrU9KunPTNhTaQRRg7J8obhSAZmgpJS6we3AqsQKkE5xMHAfsHLjklNJswJXEAnDOnUFUy3b7V+6OwMxitNR+//Yvk/SZ4lVV8sRK3heAn4PHGr70ZLxpXoN6+vLrr0ppaFN0qHAD4F/AvcQoxWN+mxvU3NMrwDftX1Oi3PrAyfYnrPOmFrEsQRRmDoHMZpynW2XjGkoq5YZvwN80XYuM/4v5QhKSqkbfINYaTI90da+WYnfxF6fzHXHEbEWUW1keBywJRMvi+2T9Ftgy2qarFRsGxAjYvMBPwa+RDQ/y+Sph2SCklIa8mwvUjqGFm4EdpV0aWMHVEmzELUWJZu0jQQ2BXYFTmFCfcwmREO7e4ExdQdVFcNeSEzv/BP4CLBXFevRklYstTv11LD9rqQtiA0i038pE5SUUmqPkcAtwMOSzgOeJhq3rUXs0zOiYGxbAaNtNzY+exwYI+l91fnaE5Tqmh8HvkB0BH67Ov5Nor5of+L9q1U1qrM58HVi08nmJqd9tlcHsN1qD5/0HmQn2ZRSaoNqGfGyRNfYdYiEZd3q+dK27ywY3nwMvPHdDZTpvAuwPjCyem/GTzHZfolYlbVMobgOAE4gesTMAszY9DG53i3pPcoRlJRSapNqGfGGpeNo4SHiZn9Fi3PLAE/VG854szFpgXO/NyjXqG1z4Ge2dyp0/Z6UCUpKKQ0SSZsAl9j+d/V4smyfWkNYrfwaOFDSa8BpRFIwL/AtYBTlesjcRmz0eHGLcxtRbgO+2YHzC127Z2WCklJKg+f3RAv5W6rHk9MHlEpQfgksSWzOd2jD8WFE3KNLBAXsCYyVdAtRLNsHfEPSrsB6NGwiWLMbiMLdTt6+oOtkH5SUUhokkj4KPGX77erxZJVuPCbpU8AKwJzAC8A1tv9eOKaVgYOALzJhCfTfgN1sX1goppWIZPJXRLLyevNrbJdcldWVMkFJKaU2mIYdg1MLkmYD5gJeqopkS8YyrulQ441zGLGKp1hfm26VUzwppdQeU7VjcF3BSDpxGl7eZ3urKb/svyfpIWC95v4mtl8FXq0jhqmwcukAelEmKCmlNEg6fMfgFZhyR915iZU0fUzYWLDdFgZmrula74ntrD0pIBOUlFIaPB27Y7Dtjw90TtLMwD7EJn1PAP9XV1xDhaTFifdoJWLvoueAa4lNF+8tGFrXygQlpZQGSYsdg/ez/VDZqCZP0jJEEzJVn3ey/XLNYXR0MaSkzxCN7V4HzmXC1gBrAWtJWqbqeZMGURbJppRSzaoRixG2Ly8YwyzAgcC2wGPA1ravLBDHOGLfnVZTYc36bKvNIU1C0kXAh4CVbb/WcHxWotndM7bXqTuubpcjKCml1AaSFgSOAVYkWqH3L5mdruFxkZUf1VLe44n6jyOBUbYnWTpbo7uYdCqsk4wAvtuYnADYfk3SGGLkKQ2yTFBSSqk9fk7c2E4kmny9TuxwvBrwGWLfmVpVS3cPA7YG7iNGcW6sO44W9rV9S+kgJuN1Bp6GGkehRLPb5WaBKaXUHisTzcV2AE4G3rS9C9GA7BpiA8HaSFoDuIfYV+ZA4HMdkpwMBTcCu1Y7PY9XTZPtQjRvS4MsR1BSSqk9ZiM6oEKMVuwFYPtdSUczcYv5OvR3YX2N2MBwQ2nAco4itR4dbCSxfcHDks4DngaGE0WysxMjZWmQZYKSUkrt8RTRVwSi58lckobbfppo3jbvgF/ZHr+lM1fL7AM8Pq1fJGkF4PbmupB2sH2vpGWJvYLWITrcvkCMhO1r++52x9CLchVPSim1gaRjiJ4Zm9u+WdKjwB+BvYGjgOVtL1YuwqlTZyIwtSRND7wNLGW71A7Hqc1yBCWllNpjD+AS4ADgy8Ao4DdEMzSI5b0drUoErgKWAjotERg25Ze8d5I2AS6x/e/q8WTZLrUzddfKBCWllNrA9nPAFyUtUD0/RdIjwLLALUOofXpbE4EO9ntiL6VbqseT00fsdpwGUSYoKaXUJpLmJxKSM6pDzxA9Ue4rFlSaWosQdUT9j1PNMkFJKaU2kLQkcDnwbyYkKMOBnYAfSFrFtkvFlybP9qMNT1cELrQ9yc7UkoYD36bGnal7RfZBSSml9jgE+Cvwhf4Dtq8DFiRGUA4pFFeadicBiw5w7nPA6Bpj6Rk5gpJSSu2xFLBB88Z7tl+VdAhZs9DRJF0ALFE9HQacI6nVfkHzEsvI0yDLBCWllNrjLWCeAc7NSWf2JEkT7A9sVT3eCriVSfcLehd4ETiuxrh6RiYoKaXUHpcA+0m6w/a9/QclfYLohXJpqcDSlNm+CbgJQNIMwH62HyobVW/JBCWllNpjZ2KPlrsk3Q88C3wYWAx4jCiWTe9BtV3AFsDDNV1vi4HOSZqZ2HTx8jpi6SXZSTallNpE0qzAFsRuxnMDLwHXASfafqVkbP2qEZ05gWdtT3LDl7QZcJ7tF2qIZRixmeHXgVmZdCFHn+3V2x1Hi7gWBI4hVvPMxITeMNP1P7adOxoPskxQUkqpB0naDtiNietkHgVG2T6tUEwHErsDP0zszzOu+TW2Vy4Q15nAqsSu1MsBrxM7HK8GfAZY3/Z5dcfV7TJBSSmlNpA0akqvsX1AHbE0k7Q9cDhwVvXxLLEaZUNiM7yNbZ8x8HdoW1xPAafY7qjpL0nPA3vZPrJK7Na2vVq1FcBY4BHbW03+u6RplTUoKaXUHvtP5tzLwJPEPj0l7AAcYftHTcf/IOlIYh+h2hMUYHbg/ALXnZLZgL9Vj+8D9oLxtTBHA4eWCqybZYKSUkptYHuSRphVTcoIop7hh7UHNcFw4KIBzp1L1M2UcAMxhdJp+xQ9RYwwQfQ8mUvScNtPA883nEuDKBOUlFKqie3XgEsk7Ut0kv18oVCuIaZzxrY49xWivqKE0cCp1bLeG4haj4nYvqH2qOBiYF9Jj9m+WdLjwI6S9gY2A54oEFPXywQlpZTq9yjwyYLXPwn4laT5gD8Q001zE6tnNgF2l7RJ/4tt19X19srq897V58YiyWHV8xKrZfYg+tocAHwZGAX8BvhJdX7bAjF1vUxQUkqpRtUOxzsDjxQM44/V569VH80OanjcR31t+WtfoTM1bD8HfFHSAtXzUyQ9QuxUfYvtTpuS6gqZoKSUUhtIeodJ29n3980YBny39qAmWKTgtQfUyTf6KrFclgnFw88QPVHuKxZUl8tlximl1AZVfULzf7B9xAqeC23nBnMtSFoc2AdYCZgDeA64Fti3ccuAmmNaErgc+LftxapjyxMrjl4HVrHtErF1s0xQUkqpB0g6DjjA9iPV48nps71NHXE1kvQZ4Hripn8eMUoxH7AW0Vl2Gdt3FYjrcmL0a93G3aklzUasenrN9tp1x9XtcoonpZQGSWNh6dSosfgUohPqUdXj1Zj8bsqlfnM9mJgyWbla8QSMX559BdFbZp0CcS0FbNCYnADYflXSIdRXo9NTMkFJKaXB8/um5/03+mEtjkGNNzbbizQ8Xriu606jEcB3G5MTiOXZksYAJ5QJi7eYeEuARnNSLqHrapM0EkoppfSeLdLwsR4xVTESWBSYBfgIsST1OWLaIk3sdQa+2Y+jzBJjiCXG+0maaGl4tdHi3sClJYLqdlmDklJKbSDpNuB022NanPsR8D3bn64/MpD0fmB3YBXgg7TeNVgF4jqH6Mq6su03G47PQvRIecn2VwvENZxoHLcQ0Un2WeDDwGLAY8AI20/WHVe3yymelFJqjyWAOwc4dy9ll/oeBmwDXAfcTItdgwsZCdwCPCzpPOBpoi3/WsQ+PSNKBGX76aqAdwuiFf/cwD3AscCJtl8pEVe3yxGUlFJqA0l/Af5ie/MW584EPmJ76doDi+s/B/zc9ugS15+cKhHYk0hG5gJeIFrz72v77pKxpXrlCEpKKbXHvsAZkhYjlqL+i5i+2AD4DLBGwdhmJpbzdpxqGfGGpeNoJGnUlF5ju9TO1F0rE5SUUmoD23+StC4xGnAgsZJnHFHL8GXb1xUM71JgTeDqgjEA45dmX2L731OzTLvmpdn99p/MuZeJvYwyQRlkmaCklFKb2D6/mupZkyhG/Q1RU3FP3bE03fxvAfaXNA9Rh/Ja8+trTAR+DyxdxdS8TLtZnfsCjWd7khWvVW+WEcAxwA/rjqkXZIKSUkptUjXx2oH4v3YcMJYYTVlA0iq2n60xnFY3/+/Sek+gOhOBRYCnGh4PCVWvlksk7QscAny+cEhdJxOUlFJqA0m7ANsDOwEXAA9Up/YGzgRGA1vXGFJH3vxtP9rwdEVin6Lnm19XLfX9NrECqZM8Cnxyiq9K0ywTlJRSao9tgL1tHyFpfIMx2zdK2h3Yr85gmhKBSVQxztrczr1mJxHTPZMkKMDniKSuYxKUaofjnYFHCofSlTJBSSml9pgfuHWAc48QvTSKkDQD0XPkQdunSlqJGNWZs9oY75u2X6wplguInjEQhcTnSHqrxUvnJZqk1U7SO0za4XY6It5htJ4mS/+lTFBSSqk9HgRWBy5vcW4E8FC94UxkH+I3/x2q578E/k0sjf4JUSfz/2qKZX9gq+rxVkRS96+m17wLvAhMaRfmdhnNpAlKH7GC50LbRRKnbpcJSkoptcfhwK8kzQicT9zQFpW0PJEc7FIwtm8BI20fXe0v8ylgc9u/lfQ8cCg1JSi2bwJugvEjO/vZLpm8TcL23qVj6EWZoKSUUhvYPl7Sh4DdiGWow4DTgbeBw2wfVTC8+YkW9xBLoMcBF1XPHwfmKBGU7S0GOidpZmLPm1YjUoNuanqyNCrUn6WrZYKSUkptYvtASUcByxA1Jy8BN7VapVKzJ4GFgWuBtYE7bT9XnVuWSFJqJ2lBoq/IisBMRFIHE+o9oL4djZuXZfdP8QxrcQwK9GfpdpmgpJRSG1WrYi4tHUeTU4GfS/o2sDywLYCkw4mpncl1Tm2nnxP1OScSm/K9DtwIrEZsD7B+jbE0Lsv+HJGw7EeMgj1FJJxrE/U8A478pPcuE5SUUuo9exDdY1cAdrV9THX888DBRFFoCSsDu9k+UtJ2wNq2d6n2whkLrAOcV0cgjcuyJZ1F1MaMaXjJk0SN0fuAMUyYIkuDJBOUlFLqMbb7iJU6BzYdX6FMROPNBvytenwfsBeA7XclHU0U75awBHDnAOfupUOb4A11maCklFIPkvQBYCVgVqLGYyKFij6fIvqdQPQ8mUvScNtPE83b5h3wK9vrH0QX28tanNsauKvecHpDJigppdRjJK1GNGabbYCXFNmUD7gY2FfSY7ZvlvQ4sKOkvYHNgCcKxATRH+YMSYsB5xJ9WuYFNiBqY9YoFFdXmyRrTiml1PUOBkyslvkYMUXR+LFoobj6a2MOqJ6PAnYEXgE2pVCbe9t/AtYlVhYdCJxAFBK/DnzZ9pUl4up2w/r6mpvjpZRS6maS3gTWsd1pq4sAkLSA7Seqx8sTy7RvsX1N4bgWJPrGfBD4DTAcuMf22yXj6lY5xZNSSr3nMWD20kEMxPYTkj4BzAk8YfuQ0jFJOoTYGmAGorHdWGI0ZQFJq9h+tmR83SineFJKqfccBOwlaaHSgTSTtJ2kp4jVMTcAD0h6SNLGBWPaBdge2An4OBPunXsTSVSpZdldLUdQUkqp92wILAA8XCUDrzed77OtuoOStD2xh9FZ1cezRDHqhsApkt61fUbdcQHbAHvbPkLS+E62tm+UtDvRwC0NskxQUkqp9zwNnFM6iBZ2AI6w/aOm43+QdCRRRFsiQZmf2GW5lUeIrrJpkGWCklJKPWZym/IVNpyBO7KeS7mW8g8CqwOtNiocAXTU7svdIhOUlFLqUZLWIJq1zQE8B1xbeGXPNcR0ztgW575C7MtTwuFEW/sZgfOJPjGLViuMdgZ2KRRXV8tlximl1GOq/WPOI276bxONx+Yhfmm9GljT9psF4toQ+BWRiPyB2O9mbuDrwCbA7jTstFxnt1tJI4HdgFmYsKPx28BhtnerK45ekglKSin1mGrJ7DbAVsCZtvskDSNGL44DjrE9skBc46bh5X22p5/yywaPpNmJnixzAy8BN9l+vs4YeklO8aSUUu/ZGNizcUVMtYHg6ZLmB34E1J6g0OGb7tl+GejI5nbdKBOUlFLqPXMCdw9w7m4Kbcpn+9ES102dKROUlFLqPQa+SutVKV8DHq43nCDpuCm9xvb364gllZcJSkop9Z7DgZOrVSmnEX1RhgPfAn5AdE0tYTVihUyj2Yiaj+cZuBdJ6kJZJJtSSj1I0j7EEtmZGg6/DRxie88yUbUmSURjuX1sn1Y6nlSPTFBSSqlHSZoTWJqoSXmBWJXyQtmoWpP0TaLd/CdLx5LqkVM8KaXUo6pk5OLScUylF4GFSweR6pMJSkop9QBJbwPL2b5V0jtMWuvRqM/2zDWFNl61xLnZ9MCCxIZ899YbUSopE5SUUuoNo4EnGh534vz+47SOaxix4/L69YaTSsoalJRSShORNL/tJwtcd3MmTVD6gJeBK6tGaalHZIKSUko9RtK7wNK2J1m2K2kEcJHtD9Qf2fgYFrX9UPX4w8AnbF9fKp5URk7xpJRSD5D0E2DW6ukwYOtqN+NmyxLLjWtXJSPnE31PFqsOLwVcIOlK4Bu2XyoRW6pfJigppdQbZgT2qB73AVu0eM27xGqZEvvwABxKtNn/XsOxi4EVgZOBA4Bt6w8rlZBTPCml1GOqXYOXtn1L6VgaSXoG2Nb2mS3ObQwcZnuB+iNLJeQISkop9Rjb05WOYQDvA94c4NwrwAdrjCUVlglKSin1IEkbEFMnMxE1KQDTEXUqy9r+aIGwbgZ2kHSJ7f/0H5Q0PbAd0FEjPqm9MkFJKaUeI2kPYB/gJeI+8E718WFgHHB8odD2BK4GHpR0EfBsFdNXgfmAVQrFlQro1GG+lFJK7bM58FtgLmJn4/Ntz0usmHkeuKdEULZvApYhdi1elyjW3RD4K9EF98YScaUyskg2pZR6jKS3gK/bvkzSesAY24tV57YHtrT9uaJBViTNTLTeL7L0OZWTIygppdR7XiOmcgAeABaRNEv1/C/AIiWCkjSdpIMk/bnh8HLAc5L2LBFTKicTlJRS6j23At+tHv8D+A8T6jsEvFUiKKIGZQfgsoZjdwOHALtI+nGRqFIRmaCklFLvORDYRNK5tt8Cfg/8VtIfgZ8BlxaKazPgp7b36z9g+9nq+W7A/xWKKxWQCUpKKfUY21cDSwNnVYe2qx5/GjgT2L5MZMwD3DfAubuAhWqMJRWWy4xTSqkH2b4DuKN6/Cbw/bIRAWBgPeDyFufWAh6sN5xUUiYoKaXUAyRtMi2vt31qu2KZjJ8Dv5E0F3AOE/qgrAV8i4n36EldLpcZp5RSD6j235lafbanb1swkyFpW2JTw3mITQ2HEb1Z9rF9ZImYUhk5gpJSSr2hyNLhaWX7qGqZ8Qhi752Xgatt3102slS3HEFJKaUeJmkG4EPAc4373xSKZTrgWGDL6lD/HkF9ROfbLW3nTatH5CqelFLqQZK+IOlS4FXgceCzkk6u9ukpZVdg0+rzgsCMxMqdkUQNyk/LhZbqllM8KaXUYyQtC1xBNEE7iKj5APgnsLek52wfUyC0rYDRtg9pOPY4MEbS+6rzYwrElQrIEZSUUuo9BwOX2V4K2J9qKsX2HsTmgT8oFNd8wPUDnLuB7IPSUzJBSSml3vMFoH+EpLmm43xg0XrDGe8hYjfjVpYBnqoxllRYTvGklFLveQWYd4BzC1TnS/g1cKCk14DTgGeIOL8FjAIOKBRXKiATlJRS6j3nAftL+ivwt+pYn6ThRCJwYaG4fgksCRwGHNpwfBixX9DoEkGlMnKZcUop9ZiqU+uVxN47TxArZv4OfBR4Elje9r8KxrcEsAIwF/ACcI3tv5eKJ5WRCUpKKfUYSYcC5wKLA6sAcwMvAdcAJ9l+rWB4KQGZoKSUUs+R9BKwge3LSseS0kByFU9KKfWe24FVSweR0uRkkWxKKfWeO4AdJK0P3EOslmnUZ3ub+sNKaYJMUFJKqfd8gyiGnR74bIvzOfefissalJRSSil1nKxBSSmllFLHyQQlpZRSSh0nE5SUUkopdZxMUFJKKaXUcf4/qxszSPvhqIIAAAAASUVORK5CYII=
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This heatmap shows that there are little to no correalation between the selected features which is needed for Naïve Bayes to work properly.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Creating-and-training-the-model">Creating and training the model<a class="anchor-link" href="#Creating-and-training-the-model">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Next step is to create the training and test data, then model and make predictions.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">df1</span><span class="o">.</span><span class="n">values</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.3</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Baseline accuracy: </span><span class="si">{:.2f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">y_train</span><span class="o">.</span><span class="n">mean</span><span class="p">()))</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">y_prob</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Naïve Bayes accuracy: </span><span class="si">{:.2}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Baseline accuracy: 0.75
Naïve Bayes accuracy: 0.77
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>With the selected features we can see an increase of 2 percentage points on predictions with these selected features.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Confusion-matrix">Confusion matrix<a class="anchor-link" href="#Confusion-matrix">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">print_conf_mtx</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;&lt;=50K&quot;</span><span class="p">,</span> <span class="s2">&quot;&gt;50K&quot;</span><span class="p">])</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>       predicted 
actual &lt;=50K &gt;50K
&lt;=50K   6149  615
&gt;50K    1474  811
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The confusion matrix shows that the model are predicting relatively good on the people below \$50K, but not that good on those with an income above \\$50K. This does also make sense to a degree since there are much more data on those with an income below \$50K.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Precision-and-recall">Precision and recall<a class="anchor-link" href="#Precision-and-recall">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Precision: </span><span class="si">{:.2f}</span><span class="s2"> Recall: </span><span class="si">{:.2f}</span><span class="s2">&quot;</span>
      <span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">precisionAndRecall</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)[</span><span class="mi">0</span><span class="p">],</span> <span class="n">precisionAndRecall</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)[</span><span class="mi">1</span><span class="p">]))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Precision: 0.57 Recall: 0.35
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The precision are pretty low with 57% which tells that the model is not really good at prediction, it only predicts right just above the half of the times. For the recall which is really bad, tells that the model is really bad at predicting the people with an actual income above \$50K.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Learning-Curve">Learning Curve<a class="anchor-link" href="#Learning-Curve">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">print_learningCurve</span><span class="p">(</span><span class="n">clf</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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f3HFFVezcWMj+fn5XH75Jfz9789y4oknM2fOtvz3v29y5523UFOzgu9//4ed53vqqceZO/dAFiy4jlBodPRf0sQyQG6ZrcDXJsdqsEWiHpfc8W/W1Gd/YM/JpeO46rTdBiW5eJ7HLbfcyBe/uDeXXHIZ0ajPbrvtwaRJk7jkkgs59tivMWfOtrz55hvsttse7Lff/gDstNPOjBs3juLi8eTm5rLVVrbT8rRp09lyy9QdmB999M88+uifk2578ME/M336jM7lL33pAA488Cvd9olGo5x11nnssMOOAHz44Qc8/fSTfO9753HMMV8FYJdddic3N5fbb7+ZY4/9GtXVmwGQm5vHBRdcTF5eXvoXa5jSxDJAnU2O61fie1Ecd3R841AqW5Ys+YTa2nV84Qt7EYlEOsfC2n33zxMKhXj11X8xZ8627LTTzjzyyB9ZvbqG3Xf/PJ/73J6ccsrp/XrNvfbahxNO+GbSbZMnV3RbnjVri6T7bb551/r//vcNwCaheHPnHsjtt9/Mf/7zemdimT59+qhKKqCJZcBiTY6JRvA2rCZUOjW7AalRKxxyueq03Ub9rbCGBjuw+YIFl7NgweWbbF+3bh0A55wzn4qKShYteoIbbriOG264ju2224ELLri424d8X5SWljF79jZ92nfChAmbrMvNzaWoqKhzecOGDbiuS1lZ931jy01NG+PWjb5GP5pYBsgpngjhPIi04a1frolFDapwyGVi6eieUTH2AX3eefPZdtvt8bzuHRxLS+0YtHl5eZx00qmcdNKprFpVwz//+QL33HMnV131I+6557dDHne84uLxeJ5HXd16JkzoShy1tTYplpT0b9TgkUJ73g+Q47idpRZvvfbAV2qgZs7cnJKSEmpqath6622YPds+iovHc8stN7Fs2VLa2lo57rgjeOih3wFQWTmFefOOZb/99mfNmtUAnS2vsmHHHXcC4Jlnnuq2Pra8/fY7DnlMQ0lLLAMQ620fKpuOt+YTrcBXKgPC4TCnnPJtfvnLnwGw666709DQwJ133kZLSzNbbrkVeXn5iGzN3XffRigUYtaszVm6dAlPPvnXznqNWMnntddeYcaMKrbZZtseX7Ouro533nk76ba8vDy23HKrtN7DlltuxX77zeXWW2+iubmZOXO25a23/stvfnMPBx10CFVV1Wmdb6TRxNJPa+tbuPCm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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This learning curve is quite surprising. There is huge error for both training and test data in the beginning and then it quickly drops down to around .23 after 2500 rows of data. Then they are really close to each other for rest of the dataset. To me this is unclear if it is a high variance or bias situation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="ROC-/-AUC">ROC / AUC<a class="anchor-link" href="#ROC-/-AUC">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">probs</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">probs</span> <span class="o">=</span> <span class="n">probs</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span>

<span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">thresholds</span> <span class="o">=</span> <span class="n">roc_curve</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">probs</span><span class="p">)</span>
<span class="n">plot_roc_curve</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">auc</span> <span class="o">=</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">probs</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;AUC: </span><span class="si">%.2f</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="n">auc</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>AUC: 0.77
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>From this plot we can see that the model is not really great from the yellow graph in the plot, also the AUC is not that great.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Model-2">Model 2<a class="anchor-link" href="#Model-2">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Features">Features<a class="anchor-link" href="#Features">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For this model, we are going to use feature forward selection algorithm to select the 10 <em>best</em> features to predict income.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df2</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">])</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">values</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.3</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">features</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">columns</span><span class="p">[</span><span class="n">getBestFeaturesCV</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>num features: 1; accuracy: 0.79; 
num features: 2; accuracy: 0.79; 
num features: 3; accuracy: 0.79; 
num features: 4; accuracy: 0.80; 
num features: 5; accuracy: 0.80; 
num features: 6; accuracy: 0.80; 
num features: 7; accuracy: 0.81; 
num features: 8; accuracy: 0.81; 
num features: 9; accuracy: 0.81; 
num features: 10; accuracy: 0.82; 
num features: 11; accuracy: 0.82; 
num features: 12; accuracy: 0.82; 
num features: 13; accuracy: 0.82; 
num features: 14; accuracy: 0.83; 
num features: 15; accuracy: 0.83; 
num features: 16; accuracy: 0.83; 
num features: 17; accuracy: 0.83; 
num features: 18; accuracy: 0.83; 
num features: 19; accuracy: 0.83; 
num features: 20; accuracy: 0.83; 
num features: 21; accuracy: 0.83; 
num features: 22; accuracy: 0.84; 
num features: 23; accuracy: 0.84; 
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>From the forward feature selection algorithm, the highest accuracy is 84% before with 22 features. The <em>best</em> features are the following:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">bestFeatures</span> <span class="o">=</span> <span class="n">features</span><span class="p">[:</span><span class="mi">22</span><span class="p">]</span>
<span class="k">for</span> <span class="n">feature</span> <span class="ow">in</span> <span class="n">bestFeatures</span><span class="p">:</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">feature</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>capital_gain
capital_loss
marital_status_Married_civ_spouse
education_Prof_school
marital_status_Never_married
age
education_Masters
education_num
education_Doctorate
workclass_Self_emp_inc
relationship_Not_in_family
occupation_Exec_managerial
education_Bachelors
relationship_Own_child
relationship_Wife
occupation_Other_service
occupation_Prof_specialty
sex
hours_per_week
occupation_Machine_op_inspct
relationship_Unmarried
education_HS_grad
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Creating-and-training-the-model">Creating and training the model<a class="anchor-link" href="#Creating-and-training-the-model">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now we can build a new model with the <em>optimal</em> features.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">bestFeatures</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.3</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Baseline accuracy: </span><span class="si">{:.2f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">y_train</span><span class="o">.</span><span class="n">mean</span><span class="p">()))</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">y_prob</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Naïve Bayes accuracy: </span><span class="si">{:.2}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">clf</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Baseline accuracy: 0.75
Naïve Bayes accuracy: 0.83
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we can see a good increase in accuracy, with 8 percentage points. This is almost as good as the accuracy on the training data, which can be a good indicator.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Confusion-matrix">Confusion matrix<a class="anchor-link" href="#Confusion-matrix">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">print_conf_mtx</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;&lt;=50K&quot;</span><span class="p">,</span> <span class="s2">&quot;&gt;50K&quot;</span><span class="p">])</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>       predicted 
actual &lt;=50K &gt;50K
&lt;=50K   6186  578
&gt;50K     935 1350
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This confusion matrix shows that the model are predicting pretty decent of the people with an income below \$50K. Still it is not that great on the people above \\$50K, but it has increased greatly from the previous model and still makes sense due to the amount of data with people above \$50K.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Precision-and-recall">Precision and recall<a class="anchor-link" href="#Precision-and-recall">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Precision: </span><span class="si">{:.2f}</span><span class="s2"> Recall: </span><span class="si">{:.2f}</span><span class="s2">&quot;</span>
      <span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">precisionAndRecall</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)[</span><span class="mi">0</span><span class="p">],</span> <span class="n">precisionAndRecall</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)[</span><span class="mi">1</span><span class="p">]))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Precision: 0.70 Recall: 0.59
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Looking at precision and recall, we now have increased in both by a great amount. 70% of the predictions are now correct of the positive predictions while of all the true positive cases the model has predicted 59% of the cases as positive.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Learning-Curve">Learning Curve<a class="anchor-link" href="#Learning-Curve">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">print_learningCurve</span><span class="p">(</span><span class="n">clf</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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i20tf3QVWmXlCo466hCuvfZGXnrpBT788D0CgSx2330vzj77fHJzczf4mvb2du655w5eeeVFGhsb2XnnXdhiiy25/fZbeeutja/9nzPnI379619y4YW/45FHHiQUCnLVVTfw3HPPsGbNakaNGsUrr8xm4sTJ3H33g7S1tfKXv8zglVdms2rVSkaNGs1RRx3HoYce0XHPI488mN1225N58+ZizFwOPvhQzjrrvI2ONVZKg0xFZBy2B9lBQL63rxH7gX9xdIJKhjFmhojkABdg22MWACcaY/7unXIgNuHsAbxhjKkWkV29GO7Etg99AOxhjHnLu2ebiOwPXAtcDgwFvgAON8Y8m0p8PcVXNprQiq8J1+g0Oyr93HAQt7Em02HgFJTh+Hp2HPvs2S+yzz778Yc/3ITjOLS0tHDWWadTUTGE88+/mMLCQj7//H889NB95OTkcsEFv+3yXtdffw0HHngI1133R77++kvuu+8uysvLOe20X23wNTfccA2vv/4Kp532K8aPn8izz/6Te++9M6nvzXVdgsH4M6EEAp2f6/3338WFF/6OpqYmNt10M5577hnmzPkIn297rr/+ZpqaWgC44IKzMWYup556OhMmTOKdd97i5puvo6ammpNOOrXjfjNnPsFRRx3HT396EkVFRfSEVMb5jMd+0A/BDvb8GttVexq2tLKXiGxnjFmaSgDGmHuBe7s4NgOYEbNvMV7JKME91wCnpxJHb1o3wagmH5VebjhI45O/w61flelQcIqHUXD0H3o0AQUCAS666BKysnIAMGYuI0aM5LLLrmLkSDuDyLbbbsdXX33Bp5/OSXivH/7wR5x55jkAbLfdDnz44fu8885/EyafRNcsX76Ml176N+eccyHTpx8NwI477szPfnYsCxcu6PZ7e/DBe3nwwbgfjbz66tvk5OR0bB9++FHsttuenc4JhUJcdNEljBw5ilDI5Z133uKTTz7m6quvZ4899gbsCrDBYJBHH32Iww8/kpISO0vZsGEjOOOMs3ESDdTZSKn8VlyLLe3sbIz5IPqAiGwLvA5cCZycvvAGJr/X6cBtrMZtbcTJ2fgGVKUGo9Gjx5Cbm9vRlVhkGnfd9QDhcJilS5ewbNlSFi5cwOLFi7q915Zbbt1pe+jQYaxalTiJJ7pmzpyPcF2X3XdflxR8Ph977LE3Cxd231vvsMOO5KCDDol7LDu7cy/ZyZM3We+cvLz8jgQM8Omnc8jKylovSe2774955pmZfPnlF/zgB7sAMHHipB5NPJBa8tkX+HNs4gEwxswRkTuAn6ctsgEsUvIBCNUsJzAitbXfleqK4wtQcPQfBk21W3l5xXr7nnjir/zlLw9TV1dHeXkF06ZtSm5uHs3NTQnvFWmnifD5fLhu4uGGia6prbU/g9LSsm5jjmfIkCFMm7ZZUueWla1/z/Ly8k7bDQ31lJWVr7dMeCSetWvXdnltT0jlN6MIWJHg+AqgLMFx5XGy83AKK3DXVtm1fTT5qDRyfAGcQbpM++zZL3LHHX/i//7vbA444GBKS2010mWX/ZZ58+b2aixDhtifQU1NDUOGDOnYH0lKva2oqIiammrC4XCnBFRVZUefRJ5Vb0llep252B5jXTkMmLdx4QwekfE+2u6jVPp89tmnlJaWcvzxJ3R8mDY1NfHZZ58SDvduR9etttoGv9/PW2+90Wn/f//7Zq/GEbHNNt+nvb2dN998rdP+l19+iaysLDbddPNejSeVks8d2O7L/8BObRNJNNOAi7Bjbc5Ib3gDl798DKGln+ns1kql0Wabbc4zz8zkrrtuY+edd2H16lX87W9/obq6ar3qr542evQY9tvvAO688zba2toYP34i//73v5g/3yTVnrJ69Sq++OLzuMcKCwuZMCG1oZc77fQDttlmW66//mpWr17FxImTePfdt3n22X/ys5+d0mO92rqSyjifB0VkGnAecETMYQfbHnRPOoMbyKJ7vLmu2+ONe0oNBvvvfxCVlSt4/vlZzJz5JEOHDmXnnXfh8MOP4sYbr2XJksWMGze+1+I5//yLyMvL45FHHqS1tZVddtmNQw+dzksv/bvba5999imeffapuMe+//0duO22u1KKxefzceONf+L+++/msccepaGhnjFjxnL++b/lsMOmd3+DNHNcN7WiqJeADsbOeOBgl77+lzHmq7RH13fUhsNuSVXV2u7P9HQ3aWZozWKanrocgIKf/glffu/Wt/YFOrFoYomeT1XVdwBUVAzv1Zj6msiSEn1xYtH6+jree+9ddt75h51KFZdd9luWL1/KQw891uMx9OTzSeZ3sKKiEJ/PqQPW+4Db0PV8erflbgDylY60i124LuHqZYMy+Sg1kOXk5HDrrTcye/YWTJ9+DDk5OXzwwXu8+eZr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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This learning curve have the same trend as for the previous model. It is still quite surprising how the graph looks. The most obvious differences is that the errors are much smaller to begin with and gets a lower error rate and the test data needs more data, around 5000 rows, for this model in order to get at a level where the training data is.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="ROC-/-AUC">ROC / AUC<a class="anchor-link" href="#ROC-/-AUC">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">probs</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">probs</span> <span class="o">=</span> <span class="n">probs</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span>

<span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">thresholds</span> <span class="o">=</span> <span class="n">roc_curve</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">probs</span><span class="p">)</span>
<span class="n">plot_roc_curve</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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BQUFrFihzJjxLP/856vcccdk2raVqnPt2LGDq64ajeoXnH32OZx77h/JyclhyZL3ueuuCXz44QfceOMt5Ob6s/v3p9RZQDoCMV55ZTXXXvsOmzbtIDc3wDnnWIJbw9gfqF+/AR07HrNH+SmndKZv3yJmz36VU0/tyowZz/Laay8zbtzNu3lfJ554Er169WHUqBFcf/01PP74s9SrVw+AKVPu5vPPP+OBBx6iffujdzt3q1aHM2nS7XTp0o3evfukXtEUkB2rYn1J6jytDRtKOf/817nggnls2rSDgoJ8br21iyW4NYz9nDp16pKXl08gECAYDPLkk4/TqVPnPYYLAZo0acLo0X/hm2/WMm/eHAA2b97Ma6+9TL9+/XczWGH69TubgQOH0qiRf5fEmKeVKCmIHgyFQjz33ArGjXuPkpIyAE4/vRV33tmNn/3MEtwaBgCV5eTs/Dapp8zJcfPFocqaZ8SorHsY5CS6vCRERUVF1btgMMj69et47LGH2b69lDPP7M3KlSvYvPkHunTpFvcsJ554Eo0bN+bddxfSr9/ZLF36AcFgkFNP7Rqzfk5ODqNHX5mgzNmBGa1ESUHC3BtuWMyDD34KQJMmdbnlls6cc05byxdoGGEqy2n63i/J3fF1piUhWO9Ifuj8YUKGa+nSJZx22im7lQUCAdq0acv48RPo0qUbb701H4CWLVvGPU9OTg4tWhzG+vXrAdi4cQMALVrEb+N3zGglTMQTWZJW7g8eLDzyyGf06XMEt93WlUMOqZeU8xqGkV20b380V155DQCbNm3k4YenEgwGufnm22jd+ggAwvlqq4v0y83NpaKivOp/cJ7b/ooZrURJQiDGihWbKSurpGPHXQluFy4caHNXhhGPnHx+6Pxh0ocHc73hwWCahgfr1y+gXbsOALRr14Gjjz6G884bypgxl/LII09x8MEHV3lY69at2+u51q37lvbt3bnCHtaGDesoLGwTs/53322iadNmVQbOb5jRSpDAPgRilJcHuf/+5dx111IKCxszf/4A6tRxXyAzWIZRDTn5VNY7PKmnDHgZzyszlOW9adNmXHHF1Ywbdy333HMnN910KyLtad78EBYseIN+/c6O2W758mVs3vwDnTu7ea8TTjiJvLw8Fi9+N+681siRF/KTnxzK1KkPp0yfVGLRg4mym6dV8zmn5cs30bPnC9x+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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">auc</span> <span class="o">=</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">probs</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;AUC: </span><span class="si">%.2f</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="n">auc</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>AUC: 0.89
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This model gives a much better ROC curve than the previous model. We can also see a significant increase for the area under curve (AUC) from 77 to 89.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Model-3">Model 3<a class="anchor-link" href="#Model-3">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Features">Features<a class="anchor-link" href="#Features">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For this model the features from the forward feature selection algorithm will be used. The model of choice will be logisitc regression.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df3</span> <span class="o">=</span> <span class="n">df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">])</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">df3</span><span class="o">.</span><span class="n">values</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.3</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">features3</span> <span class="o">=</span> <span class="n">df3</span><span class="o">.</span><span class="n">columns</span><span class="p">[</span><span class="n">getBestFeaturesLogCV</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>num features: 1; accuracy: 0.79;
num features: 2; accuracy: 0.80;
num features: 3; accuracy: 0.81;
num features: 4; accuracy: 0.81;
num features: 5; accuracy: 0.84;
num features: 6; accuracy: 0.84;
num features: 7; accuracy: 0.84;
num features: 8; accuracy: 0.84;
num features: 9; accuracy: 0.85;
num features: 10; accuracy: 0.85;
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>From the forward feature selection algorithm the best accuracy is at 85% with 9 features. The selected features are:</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">bestFeatures3</span> <span class="o">=</span> <span class="n">features3</span><span class="p">[:</span><span class="mi">9</span><span class="p">]</span>
<span class="k">for</span> <span class="n">feature</span> <span class="ow">in</span> <span class="n">bestFeatures3</span><span class="p">:</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">feature</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>capital_gain
capital_loss
education_Masters
marital_status_Married_civ_spouse
education_num
occupation_Exec_managerial
workclass_Self_emp_not_inc
occupation_Farming_fishing
occupation_Other_service
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Creating-and-training-the-model">Creating and training the model<a class="anchor-link" href="#Creating-and-training-the-model">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="n">bestFeatures3</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;label&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.3</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Baseline accuracy: </span><span class="si">{:.2f}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">y_train</span><span class="o">.</span><span class="n">mean</span><span class="p">()))</span>
<span class="n">clf3</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">()</span>
<span class="n">clf3</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">clf3</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">y_prob</span> <span class="o">=</span> <span class="n">clf3</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Logistic regression accuracy: </span><span class="si">{:.2}</span><span class="s2">&quot;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">clf3</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Baseline accuracy: 0.75
Logistic regression accuracy: 0.84
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For the logistic model, we have an increase of 9 percent points in accuracy with <em>only</em> 9 features.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Confusion-matrix">Confusion matrix<a class="anchor-link" href="#Confusion-matrix">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">print_conf_mtx</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;&lt;=50K&quot;</span><span class="p">,</span> <span class="s2">&quot;&gt;50K&quot;</span><span class="p">])</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>       predicted 
actual &lt;=50K &gt;50K
&lt;=50K   6258  506
&gt;50K     961 1324
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The confusion matrix shows that we have pretty decent prediction rate for those with an income below \$50K while it has not that great on the people below \\$50K.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Precision-and-recall">Precision and recall<a class="anchor-link" href="#Precision-and-recall">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Precision: </span><span class="si">{:.2f}</span><span class="s2"> Recall: </span><span class="si">{:.2f}</span><span class="s2">&quot;</span>
      <span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">precisionAndRecall</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)[</span><span class="mi">0</span><span class="p">],</span> <span class="n">precisionAndRecall</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">y_pred</span><span class="p">)[</span><span class="mi">1</span><span class="p">]))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Precision: 0.72 Recall: 0.58
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For this model we now have a precision and recall at 72% and 58% respectively, which is an increase of precision by 2 percentage points and a decrease by 1 percentage point of recall compared to the best Naïve Bayes model.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Learning-curve">Learning curve<a class="anchor-link" href="#Learning-curve">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">print_learningCurve</span><span class="p">(</span><span class="n">clf3</span><span class="p">,</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As for the previous models it looks like we have a learning curve that shows that the model needs to have around 2500 rows of data to stabilize the error rate. This curve shows a trend towards a high bias situation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="ROC-/-AUC">ROC / AUC<a class="anchor-link" href="#ROC-/-AUC">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[48]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">probs</span> <span class="o">=</span> <span class="n">clf3</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
<span class="n">probs</span> <span class="o">=</span> <span class="n">probs</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">]</span>

<span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">,</span> <span class="n">thresholds</span> <span class="o">=</span> <span class="n">roc_curve</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">probs</span><span class="p">)</span>
<span class="n">plot_roc_curve</span><span class="p">(</span><span class="n">fpr</span><span class="p">,</span> <span class="n">tpr</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[49]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">auc</span> <span class="o">=</span> <span class="n">roc_auc_score</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span> <span class="n">probs</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;AUC: </span><span class="si">%.2f</span><span class="s1">&#39;</span> <span class="o">%</span> <span class="n">auc</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>AUC: 0.88
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For this ROC curve we can see a similar trend to the model 2 curve. It is pretty good and we have a good AUC at 88.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Conclusion">Conclusion<a class="anchor-link" href="#Conclusion">&#182;</a></h1><hr>
<p>Selecting features in blindness is not easy for this project. With help of a heatmap I tried a lot of different features with low correaltion to beat baseline accuracy, the best result ended with an increase of 2 percentage points in accuracy. This model did not do really well on predictions on test data which was somewhat surprising with that accuracy.</p>
<p>As for the second model, using forward feature selection made a great increase in number of features and results. These predictions gave pretty good results on the cases where income is below \$50K which also is desribed earlier in the report. The biggest challenge with this model is the predict correctly for people with an income above \\$50K.</p>
<p>For the last model I chose to use Logistic Regression in an attempt to do better than Naïve Bayes. Again by using forward feature selection, I ended up only using 9 features. By running the forward feature selection algorithm with cross validation it took a lot of computing resources, so I decided to stop the algorithm at 20 features. I could not see any accuracy increase after 9 features while it might be better with 20+ features.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

    </div>
</div>
</div>

</div>
    </div>
  </div>
</body>

 


</html>
