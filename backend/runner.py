# backend/runner.py
import threading
import logging
from collectors import ENT_Kindex  # import your collector modules here
# from collectors import kp_script, solar_script, proton_script, mag_script  # uncomment as needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_collectors():
    threads = []

    # ENT K-index collector
    ent_thread = threading.Thread(target=ENT_Kindex.run_ent_kindex, kwargs={'update_interval_minutes': 10})
    ent_thread.daemon = True
    threads.append(ent_thread)

    # Add other collectors in the same way
    # Example:
    # kp_thread = threading.Thread(target=kp_script.run_kp_collector, kwargs={'update_interval_minutes': 10})
    # kp_thread.daemon = True
    # threads.append(kp_thread)

    # Start all threads
    for t in threads:
        t.start()
        logging.info(f"Started thread {t.name}")

    # Keep the main thread alive while collectors run
    try:
        while True:
            for t in threads:
                if not t.is_alive():
                    logging.warning(f"Thread {t.name} has stopped. Restarting...")
                    t.start()
            # Sleep for a bit before next check
            import time
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Stopping all collectors...")

if __name__ == "__main__":
    start_collectors()
