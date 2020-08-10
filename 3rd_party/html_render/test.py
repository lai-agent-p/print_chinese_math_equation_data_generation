import execjs
import pdb
ctx = execjs.compile("""
    function generate(latex_list, ids){
        require('mathjax').init({
            loader: { load: ['input/tex', 'output/svg'] }
        }).then((MathJax) => {
            const fs = require('fs');
            var dict = {};
            const svg = MathJax.startup.adaptor.outerHTML(MathJax.tex2svg(latex_list[0], { display: true }));
            dict[ids[0]] = {'label':latex_list[0], 'svg':svg};
            let data = JSON.stringify(dict);
            fs.writeFileSync('test.json', data);
            return svg; 
        })
    return 1
    }
    """)
# pdb.set_trace()
out = ctx.call("generate", ['5'], ['test'])
print(out)
