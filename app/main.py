from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
from markupsafe import Markup
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Allow Jinja to use newline to <br> conversion
def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

# Initialize QA chain once (avoid reloading each request)
qa_chain = create_qa_chain()

# ==============================
# ROUTE: Home Chat Page
# ==============================
@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

        try:
            response = qa_chain.invoke({"query": user_input})
            result = response.get("result", "No response")

            messages.append({"role": "assistant", "content": result})
            session["messages"] = messages

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return render_template("index.html", messages=session["messages"], error=error_msg)

        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))


# ==============================
# ROUTE: Clear Chat
# ==============================
@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))


# ==============================
# ROUTE: Health Check
# ==============================
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


# ==============================
# ROUTE: Recommendations
# ==============================
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    if request.method == "GET":
        return render_template("recommend.html")

    # Detect JSON vs form submission
    if request.is_json:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request"}), 400
        user_query = data["query"]
    else:
        user_query = request.form.get("query")
        if not user_query:
            return render_template("recommend.html", error="Please enter a query!")

    try:
        response = qa_chain.invoke({"query": user_query})
        result_text = response.get("result", "")

        recommendations = []

        # Only keep clean "name - url" lines
        for line in result_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Skip lines without URLs
            if "http" not in line:
                continue
            # Split name and URL
            if "-" in line:
                name, url = line.split("-", 1)
                name, url = name.strip(), url.strip()
                if name and url:
                    recommendations.append({"name": name, "url": url})

        # Limit to top 10 results
        recommendations = recommendations[:10]

        # Return JSON if it's an API request
        if request.is_json:
            return jsonify({"recommendations": recommendations}), 200

        # Otherwise render HTML form
        return render_template("recommend.html", recommendations=recommendations, query=user_query)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        else:
            return render_template("recommend.html", error=str(e))


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
