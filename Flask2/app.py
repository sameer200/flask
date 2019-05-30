from flask import Flask, redirect, url_for, render_template, request
import Main

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('askgrid'))


@app.route('/askgrid')
def askgrid():
    return render_template("askgrid.html")

@app.route('/script', methods=['POST' , 'GET'])
def script():

    grid = request.form['grid']
    gridint = int(grid)
    Main.Process(gridint)
    return render_template("plot.html" , url ='/static/images/plot.png')




if __name__ == '__main__':
    app.run(debug=True)

