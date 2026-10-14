"""
Flask live dashboard — serves the same page as reports/dashboard.html with
fresh broker data on every request, plus a JSON API.

Bound to 127.0.0.1 by default. Listening on another interface (e.g. a
Tailscale IP) requires DASHBOARD_TOKEN: pass ?token=... once; it is then kept
in an HttpOnly cookie. Read-only: there are no endpoints that change state.
"""
import hmac
import os
import sys

from flask import Flask, Response, jsonify, request, make_response

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOT_DIR)


def create_app(config=None, dp=None, token=None):
    from dashboard.render import collect, render_html

    app = Flask(__name__, static_folder="assets", static_url_path="/assets")
    if config is None:
        import yaml
        with open(os.path.join(BOT_DIR, "config.yaml")) as f:
            config = yaml.safe_load(f)

    def _authorized():
        if not token:
            return True
        given = request.args.get("token") or request.cookies.get("fb_token") or ""
        return hmac.compare_digest(given, token)

    def _data():
        account = positions = None
        if dp is not None and getattr(dp, "trading_client", None):
            try:
                account, positions = dp.get_account(), dp.get_positions()
            except Exception:
                pass
        return collect(config, account, positions, trading_client=getattr(dp, "trading_client", None))

    @app.before_request
    def _guard():
        if not _authorized():
            return Response("Unauthorized — append ?token=<DASHBOARD_TOKEN>", status=401)

    @app.route("/")
    def index():
        page = render_html(_data()).replace("</body>", """<script>
(function(){
  const key='fb-dashboard-view';
  function save(){try{sessionStorage.setItem(key,JSON.stringify({y:scrollY,open:Array.from(document.querySelectorAll('details')).map(d=>d.open)}))}catch(e){}}
  try{const old=JSON.parse(sessionStorage.getItem(key)||'{}');
    document.querySelectorAll('details').forEach((d,i)=>d.open=!!(old.open||[])[i]);
    if(typeof old.y==='number')requestAnimationFrame(()=>scrollTo(0,old.y));}catch(e){}
  document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',save));
  addEventListener('beforeunload',save);
  setInterval(()=>{if(document.visibilityState==='visible'){save();location.reload()}},60000);
  const age=document.getElementById('updated-ago');
  if(age){const tick=()=>{const min=Math.max(0,Math.floor((Date.now()/1000-Number(age.dataset.updated))/60));age.textContent=`updated ${min} min ago`};tick();setInterval(tick,60000)}
})();
</script></body>""", 1)
        resp = make_response(page)
        if token and request.args.get("token"):
            resp.set_cookie("fb_token", request.args["token"], httponly=True, samesite="Strict")
        return resp

    @app.route("/api/data")
    def api_data():
        d = _data()
        d.pop("latest_report", None)
        return jsonify(d)

    return app
