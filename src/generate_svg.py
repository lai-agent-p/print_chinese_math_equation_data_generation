import execjs


def generate_svg(label, path):
    ctx = execjs.compile("""
    process.env.NODE_PATH = "/home/agent_p/hand_chinese_math_equation_data_generator/3rd_party/html_render"
    function generate(label, path){
        require('mathjax').init({
            loader: { load: ['input/tex', 'output/svg'] }
        }).then((MathJax) => {
            const fs = require('fs');
            const svg = MathJax.startup.adaptor.outerHTML(MathJax.tex2svg(label, { display: true }));
            fs.writeFileSync(path, svg);
        });
    }
    """)
    ctx.call("generate", label, path)
    
    
# generate_svg('\\permil', 'temp.svg')
# \\textperthousand
# \\romannumeral
# \\textcircled
# \\RomanNumeralCaps