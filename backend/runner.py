import threading
import logging
import time
from collectors import ENT_Kindex, EthTEC
import os
import sys
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPDATE_INTERVAL_MINUTES = 10

def run_ent_kindex():
    """Run ENT K-index in a loop."""
    while True:
        try:
            ENT_Kindex.run_ent_kindex(update_interval_minutes_param=UPDATE_INTERVAL_MINUTES)
        except Exception as e:
            logging.error(f"ENT_Kindex error: {e}")
        time.sleep(60) 

def run_ethetec(stop_event):
    """Run EthTEC continuously until stopped."""
    try:
        EthTEC.main(stop_event=stop_event)
    except Exception as e:
        logging.error(f"EthTEC error: {e}")

def run_s4pi():
    """Run S4_Pi.py continuously."""
    script_path = os.path.join(os.path.dirname(__file__), "S4_Pi.py")
    if not os.path.exists(script_path):
        logging.error(f"S4_Pi.py not found at {script_path}")
        return

    while True:
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"S4_Pi.py exited with error: {e}")
        time.sleep(180) 

def start_collectors():
    threads = []

    #ENT K-index 
    ent_thread = threading.Thread(target=run_ent_kindex, name="ENT_Kindex_Thread", daemon=True)
    threads.append(ent_thread)

    #EthTEC 
    ethetec_stop = threading.Event()
    ethetec_thread = threading.Thread(target=run_ethetec, args=(ethetec_stop,), name="EthTEC_Thread", daemon=True)
    threads.append(ethetec_thread)

    #S4_Pi 
    s4pi_thread = threading.Thread(target=run_s4pi, name="S4_Pi_Thread", daemon=True)
    threads.append(s4pi_thread)


    for t in threads:
        t.start()
        logging.info(f"Started collector thread: {t.name}")

    #Monitor 
    try:
        while True:
            for t in threads:
                if not t.is_alive():
                    logging.warning(f"Thread {t.name} stopped. Restarting...")
                    t.start()
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping all collectors...")
        ethetec_stop.set()  

if __name__ == "__main__":
    start_collectors()
