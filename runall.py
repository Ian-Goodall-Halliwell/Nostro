from datetime import datetime, timedelta, timezone


import pytz
import time
import line_profiler

from sqlalchemy import true
import limits
import random
import optuna
import socket
import struct
import time
from create_orders import mainfunc
import win32api

# List of servers in order of attempt of fetching

import threading


'''
Returns the epoch time fetched from the NTP server passed as argument.
Returns none if the request is timed out (5 seconds).
'''

def update_orders():
    t = 0
    while True:
        t += 1
        if t >= 10:
            t=0
            #sync_time()
            mainfunc(False,False,False,True)
            # result = subprocess.run(['python','create_orders.py','False','False','False','True'], capture_output=True)
            # print(result.stdout.decode("utf-8") )
            # print(result.stderr.decode("utf-8") )
            print("Trades updated")
        time.sleep(1)
        
def gettime_ntp(addr='time.nist.gov'):
    # http://code.activestate.com/recipes/117211-simple-very-sntp-client/
    TIME1970 = 2208988800      # Thanks to F.Lundh
    client = socket.socket( socket.AF_INET, socket.SOCK_DGRAM )
    data = bytes('\x1b' + 47 * '\0','utf-8')
    try:
        # Timing out the connection after 5 seconds, if no response received
        client.settimeout(5.0)
        client.sendto( data, (addr, 123))
        data, address = client.recvfrom( 1024 )
        if data:
            epoch_time = struct.unpack( '!12I', data )[10]
            epoch_time -= TIME1970
            return epoch_time
    except socket.timeout:
        return None
    
def sync_time():
    epoch_time = gettime_ntp('time.windows.com')
    if epoch_time is not None:
        # SetSystemTime takes time as argument in UTC time. UTC time is obtained using utcfromtimestamp()
        utcTime = datetime.utcfromtimestamp(epoch_time)
        win32api.SetSystemTime(utcTime.year, utcTime.month, utcTime.weekday(), utcTime.day, utcTime.hour, utcTime.minute, utcTime.second, 0)
        # Local time is obtained using fromtimestamp()
        localTime = datetime.fromtimestamp(epoch_time)
        print("Time updated to: " + localTime.strftime("%Y-%m-%d %H:%M") + " from " + 'time.windows.com')
    else:
        print("Could not find time from " + 'time.windows.com')
def rounded_to_the_last_epoch_1m(now):
    rounded = now - (now - datetime.min.replace(tzinfo=timezone.utc)) % timedelta(minutes=1)
    return rounded.replace(tzinfo=None)

if __name__ == "__main__":
    
    study_name2="backtest"
    storage_name2 = "sqlite:///backtest.db".format(study_name2)
    study2 = optuna.create_study(study_name=study_name2,storage=storage_name2,load_if_exists=True,direction='maximize')
    bestparams2 = study2.best_params
    
    __,_ = limits.create(bestparams2['index modifier variable'], l=bestparams2['low modifier variable'], h=bestparams2['high modifier variable'],std=bestparams2['std'],keyt=False,
                         quant=bestparams2['std'],
                         iqh=bestparams2['high quantile modifier variable'], iql=bestparams2['low quantile modifier variable'])
    
    # __,_ = limits.create(bestparams2['index modifier variable'], l=bestparams2['low modifier variable'], h=bestparams2['high modifier variable'],std=bestparams2['std'],keyt=False,
    #                      )
    
    
    counter = 0
    
    intn = random.randint(1,1000000)
    time_min = rounded_to_the_last_epoch_1m(datetime.now(pytz.utc)).minute

    t = 0
    thr = threading.Thread(target=update_orders, args=(), kwargs={})
    thr.start() # Will run "foo"

    while True:
        
        if time_min != rounded_to_the_last_epoch_1m(datetime.now(pytz.utc)).minute:
            if datetime.now(pytz.utc).minute < 60:


                time_min = rounded_to_the_last_epoch_1m(datetime.now(pytz.utc)).minute
                p = datetime.now(pytz.utc)
                pn = rounded_to_the_last_epoch_1m(datetime.now(pytz.utc))
                
                
                
                print("Starting the UTC {} trade".format(pn))
                
                
                
                print('create orders')
                
                mainfunc(True,True,True)
                # result = subprocess.run(['python','create_orders.py','True','True','True'], capture_output=True)
                # print(result.stdout.decode("utf-8") )
                # print(result.stderr.decode("utf-8") )
                # with open('logs/log_internal_{}.txt'.format(intn), 'a',encoding="utf-8") as log:
                #         log.write(result.stdout.decode("utf-8"))
                #         log.write(result.stderr.decode("utf-8"))
                print("Done the UTC {} trade".format(pn))
                
                

                timdur = datetime.now(pytz.utc) - p
                print("Time elapsed: {}".format(timdur))

                counter += 1
                
            else:
                print('Not there yet: {}'.format(60 - datetime.now(pytz.utc).minute))

        else:
            time.sleep(1)
    print('done')
    
            
        
    