# assets/

Static assets used by the Streamlit app (e.g. `Logo.png` for the sidebar).

`Logo.png` was referenced by the app but was never actually present in the
original repository, so `app/main.py` now checks for its existence before
trying to display it instead of crashing. Drop your logo file here as
`assets/Logo.png` to have it appear in the sidebar.
