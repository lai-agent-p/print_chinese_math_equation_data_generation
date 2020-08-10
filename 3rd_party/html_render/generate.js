var res
mjAPI = require('mathjax');

mjAPI.config ({
  MathJax : {
      SVG : {
          scale: 120,
          font : "STIX-Web",
          undefinedFamily:"'Arial Unicode MS',serif",
          addMMLclasses: true
      }
  },
  displayErrors : true,
  displayMessages : false
});

mjAPI.start();
mjAPI.typeset({
  math: yourMath,
  format: "MathML",
  svg:true,
}

var gene = function(latex_list, ids){var promise =  new Promise(function(resolve, reject){
    require('mathjax').init({
        loader: { load: ['input/tex', 'output/svg'] }
    }).then((MathJax) => {
        resolve(MathJax.startup.adaptor.outerHTML(MathJax.tex2svg(latex_list[0], { display: true })));  
    });    
})
return promise
};

function test(temp) {console.log(temp);};
gene(['5'], ['test']).then(val =>{
    res = val; 
});
console.log(res)
// async function yes() {
  // const result = await gene(['5'], ['test']);
  // console.log(result); // --> 'done!';
// }



// please put all latex in the list
// all generated svg will be in generated_svg.json
// run with  node generate.js
// project already includes required packages, no need to install. 
// var path = require("/home/agent_p/hand_chinese_math_equation_data_generator/data/train_labels.json");
// var labels = [];
// var ids = ['test'];
// console.log(path)
// for (let [key, value] of Object.entries(path)) {
    // labels.push(value['label']);
    // ids.push(key)
// }
// var list = ["\\frac { - b \\pm \\sqrt { b ^ { 2 } - 4 a c } } { 2 a }"];
// var temp = "";
// temp = generate(list, ids);
// console.log(temp);
