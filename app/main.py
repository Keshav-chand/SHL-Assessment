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
            # Fixed: use .run() instead of .invoke()
            result = qa_chain.run(user_input)

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

    user_query = None
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
        # Fixed: use .run() instead of .invoke()
        result_text = qa_chain.run(user_query).strip()

        # Try to parse "name - url" lines for structured recommendations
        recommendations = []
        for line in result_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Only consider lines with a URL
            if "http" in line and "-" in line:
                name, url = line.split("-", 1)
                name, url = name.strip(), url.strip()
                if name and url:
                    recommendations.append({"name": name, "url": url})

        # Limit to top 10
        recommendations = recommendations[:10]

        # If JSON request, return JSON
        if request.is_json:
            if recommendations:
                return jsonify({"recommendations": recommendations}), 200
            else:
                return jsonify({"answer": result_text}), 200

        # Render HTML template
        return render_template(
            "recommend.html",
            query=user_query,
            recommendations=recommendations if recommendations else None,
            answer=result_text if not recommendations else None
        )

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
