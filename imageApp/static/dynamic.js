function spoiler() {
  document.getElementById("spoilers").style.display="block";
}

function dropdown() {
  document.getElementsByClassName("onclickdropdown").style.display="block";
}

function comment() {
  document.getElementById("comments").style.display="block";
}

// const rotatingHobbies = [
//   {text: "programming", color: "green"},
//   {text: "reading", color: "blue"},
//   {text: "volunteering", color: "red"},
//   {text: "salsa dancing", color: "orange"},
//   {text: "having fun", color: "white"}
// ]

// function typeSentence(sentence) {
//   i = 0
//   while (i < sentence.length) {
//     document.getElementById("sentence").innerHTML += sentence.charAt(i);
//     i++
//     setTimeout(typeSentence, 100)
//   }
//   return;
// }

// function rotate () {
//   var i = 0;
//   while (true) {
//     // updateFontColor(rotatingHobbies[i].text, rotatingHobbies[i].color)
//     typeSentence(rotatingHobbies[i].text);
//     setTimeout(1500);
//     deleteSentence(rotatingHobbies[i].text);
//     setTimeout(500);
//     i++
//     if(i >= rotatingHobbies.length) {i = 0;}
//   }
// }

// function deleteSentence(sentence) {
//   i = sentence.length - 1
//   while (i < sentence.length) {
//     document.getElementById("sentence").innerHTML -= sentence.charAt(i);
//     i--
//     setTimeout(typeSentence, 100)
//   }
//   return;
// }

// // function updateFontColor(eleRef, color) {
// //   $(eleRef).css('color', color);
// // }
//        <!-- <div class="typing-container" >
// {/* <span id="sentence" class="sentence" onload="typeSentence(rotatingHobbies.text)">I love </span>
// <span class="input-cursor"></span>
// </div> --> */}