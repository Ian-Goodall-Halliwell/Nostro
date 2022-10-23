from binance.client import Client
import time
import csv
import os
import line_profiler

import queue
import shutil

from binance.enums import HistoricalKlinesType

try:
    import checker
except:
    from data_pulling import checker
from binance.client import Client
import dateparser
import pytz
from datetime import datetime, timedelta
import threading
from threading import Lock


s_print_lock = Lock()
try:
    from data_pulling import dump_bin
except:
    import dump_bin


def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


def date_to_milliseconds1(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


def get_exchange_info():
    # time.sleep(1)
    client9 = Client(
        "DRxgIOPKpRZXSIhYHafd5CMdVIapwuaoG4AIjO8mzBYD7bTQ06pPXFXOf76ECwOE",
        "PYimvftBK6hyHGIPIfhIcEXW6vuCPwnpxqFPTWsUNkH8ZWbUzlV2oiZRbFrsl1uz",
    )
    dd = client9.get_exchange_info()
    return dd


def getstate(outlist, exchangeinfo, tokennames=["BUSD"]):
    t = False
    if t == True:
        currlist = [
            "BTCBUSD",
            "ETHBUSD",
            "BNBBUSD",
            "ADABUSD",
            "SOLBUSD",
            "DOTBUSD",
            "MATICBUSD",
            "AVAXBUSD",
            "TRXBUSD",
            "NEARBUSD",
            "ATOMBUSD",
            "ALGOBUSD",
            "VETBUSD",
            "ICPBUSD",
            "EOSBUSD",
            "FTMBUSD",
            "WAVESBUSD",
            "OPBUSD",
            "WAXPBUSD",
        ]
        return currlist
    currlist = []
    for ab in exchangeinfo["symbols"]:

        if ab["quoteAsset"] == tokennames[0]:
            if not ab["symbol"] in outlist:
                if not "BEAR" in ab["symbol"]:
                    if not "BULL" in ab["symbol"]:
                        if not "UP" in ab["symbol"]:
                            if not "DOWN" in ab["symbol"]:
                                client9 = Client(
                                    "DRxgIOPKpRZXSIhYHafd5CMdVIapwuaoG4AIjO8mzBYD7bTQ06pPXFXOf76ECwOE",
                                    "PYimvftBK6hyHGIPIfhIcEXW6vuCPwnpxqFPTWsUNkH8ZWbUzlV2oiZRbFrsl1uz",
                                )
                                sm = client9.get_klines(
                                    symbol=ab["symbol"],
                                    interval=Client.KLINE_INTERVAL_1DAY,
                                    limit=30,
                                )
                                try:
                                    sm.pop(-1)
                                    vol = [float(x[5]) for x in sm]
                                    v = [float(x[4]) for x in sm]
                                    vol = sum(vol) / len(vol)
                                    v = sum(v) / len(v)
                                except:
                                    continue
                                vv = vol * v
                                if vv > 5000000:
                                    currlist.append(ab["symbol"])

        if "DEFI" in ab["symbol"]:

            currlist.append(ab["symbol"])
        # if ab['quoteAsset'] == tokennames[1]:
        #     if not ab['symbol'] in outlist:
        #         if not "BEAR" in ab['symbol']:
        #             if not "BULL" in ab['symbol']:
        #                 currlist.append(ab['symbol'])
    print("number of tokens:", len(currlist))
    return currlist


def download1(start, end, interval, q, pth, type1, client):

    # if withcgdata == True:
    pq = None
    try:
        token = q.get()

    except:
        print("e")
        return

    if token == None:
        return
    if type1 == "1min":
        try:
            order = 0
        except:
            print("wonko")
    if type1 == "5min" or "30min":
        order = 0
    if type1 == "day":
        order = pq
    if type1 == "6h":
        order = pq
    if type1 == "1h":
        order = pq
    if type1 == "15m":
        order = pq
    # print(order)
    # print("starting {}".format(token))

    cval = None
    try:

        klines = client.get_historical_klines(
            token, interval, start, end_str=end, klines_type=HistoricalKlinesType.SPOT
        )  # ,barorder=order)

    except Exception as e:
        print(e)
        if e == {"error": "Could not find coin with the given id"}:

            return
        cm = 0
        b = False
        while b == False:
            if cm > 10:
                pq.put(order)
                return
            try:

                klines = client.get_historical_klines(
                    token, interval, start, end, klines_type=HistoricalKlinesType.SPOT
                )  # ,barorder=order)
                b = True
            except Exception as e:
                print("retrying", e)
                cm += 1
    if klines == []:
        return

    cnct = 0
    with open(os.path.join(pth, "{}.csv".format(token)), "w", newline="") as f:
        writerc = csv.writer(f)
        dic = ["date", "open", "high", "low", "close", "volume", "symbol", "factor"]
        writerc.writerow(dic)
        gct = 0
        it = 0
        prevdate = 0
        templist = []
        o = klines[-1]
        for a in klines:
            if (
                datetime.fromtimestamp(a[0] / 1000.0, tz=pytz.utc).strftime("%d")
                != prevdate
            ):
                cnct = cnct + 1
                prevdate = datetime.fromtimestamp(a[0] / 1000.0, tz=pytz.utc).strftime(
                    "%d"
                )

            if type1 == "day":
                dd = datetime.fromtimestamp(a[0] / 1000.0, tz=pytz.utc).strftime(
                    "%Y-%m-%d"
                )
            else:
                dd = datetime.fromtimestamp(a[0] / 1000.0, tz=pytz.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            b = [
                dd,
                a[1],
                a[2],
                a[3],
                a[4],
                float(a[5]) * (float(a[1]) + float(a[4])),
                token,
                1,
            ]
            writerc.writerow(b)
            gct = gct + 1
            if gct == 1000:
                it = it + 1
                gct = 0

        writerc.writerows(templist)


def startdownload_5m(start, end, dir, app=False):

    if not os.path.exists(dir):
        os.mkdir(dir)

    exchg = get_exchange_info()
    if app == True:
        currlist = []
        for popl in os.listdir(
            "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/5m-unprocessed"
        ):
            if popl.split(".")[0] == "CCI":
                continue
            currlist.append(popl.split(".")[0])
    else:

        currlist = getstate([], exchg)
    q = queue.SimpleQueue()
    for item in currlist:
        q.put(item)

    interval = Client.KLINE_INTERVAL_5MINUTE

    idc = 0
    for a in range((len(currlist) // 14) + 1):
        if q.empty() == True:
            return
        # t1 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client1),daemon=False)
        # t2 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client2),daemon=False)
        # t3 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client3),daemon=False)
        # t4 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client4),daemon=False)
        # t5 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client5),daemon=False)
        # t6 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client6),daemon=False)
        # t7 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client7),daemon=False)
        # t8 = threading.Thread(target=download1, args=(start,end,interval,q,dir,'5m',client8),daemon=False)
        t9 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client9),
            daemon=False,
        )
        t10 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client10),
            daemon=False,
        )
        t11 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client11),
            daemon=False,
        )
        t12 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client12),
            daemon=False,
        )
        t13 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client13),
            daemon=False,
        )
        t14 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client14),
            daemon=False,
        )
        t15 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client15),
            daemon=False,
        )
        t16 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client16),
            daemon=False,
        )
        t17 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client17),
            daemon=False,
        )
        t18 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client18),
            daemon=False,
        )
        t19 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client19),
            daemon=False,
        )
        t20 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client20),
            daemon=False,
        )
        t21 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client21),
            daemon=False,
        )
        t22 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "5m", client22),
            daemon=False,
        )

        # thrds = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22]

        thrds = [t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22]
        if q.qsize() < 14:
            for en, thrd in enumerate(range(q.qsize())):
                thrds[en].start()
            thrds[en].join()
        else:
            # t1.start()
            # # time.sleep(5)
            # t2.start()
            # # time.sleep(5)
            # t3.start()
            # # time.sleep(5)
            # t4.start()
            # # time.sleep(5)
            # t5.start()
            # # time.sleep(5)
            # t6.start()
            # # time.sleep(5)
            # t7.start()
            # # time.sleep(5)
            # t8.start()
            # time.sleep(5)
            t9.start()
            # time.sleep(5)
            t10.start()
            # time.sleep(5)
            t11.start()
            # time.sleep(5)
            t12.start()
            # time.sleep(5)
            t13.start()
            # time.sleep(5)
            t14.start()
            # time.sleep(5)
            t15.start()
            # time.sleep(5)
            t16.start()
            # time.sleep(5)
            t17.start()
            # time.sleep(5)
            t18.start()
            # time.sleep(5)
            t19.start()
            # time.sleep(5)
            t20.start()
            # time.sleep(5)
            t21.start()
            # time.sleep(5)
            t22.start()
            # time.sleep(5)

            # t1.join()
            # t2.join()
            # t3.join()
            # t4.join()
            # t5.join()
            # t6.join()
            # t7.join()
            # t8.join()
            t9.join()
            t10.join()
            t11.join()
            t12.join()
            t13.join()
            t14.join()
            t15.join()
            t16.join()
            t17.join()
            t18.join()
            t19.join()
            t20.join()
            t21.join()
            t22.join()
            idc += 1
            if idc == 3:
                idc = 0
                time.sleep(1)


def startdownload_1m(start, end, dir, app=False):

    if not os.path.exists(dir):
        os.mkdir(dir)

    exchg = get_exchange_info()
    if app == True:
        currlist = []
        for popl in os.listdir(
            "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-unprocessed"
        ):
            if popl.split(".")[0] == "CCI":
                continue
            currlist.append(popl.split(".")[0])
    else:

        currlist = getstate([], exchg)
    q = queue.SimpleQueue()
    for item in currlist:
        q.put(item)

    interval = Client.KLINE_INTERVAL_1MINUTE

    idc = 0
    for a in range((len(currlist) // 22) + 2):
        if q.empty() == True:
            return
        t1 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client1),
            daemon=False,
        )
        t2 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client2),
            daemon=False,
        )
        t3 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client3),
            daemon=False,
        )
        t4 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client4),
            daemon=False,
        )
        t5 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client5),
            daemon=False,
        )
        t6 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client6),
            daemon=False,
        )
        t7 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client7),
            daemon=False,
        )
        t8 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client8),
            daemon=False,
        )
        t9 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client9),
            daemon=False,
        )
        t10 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client10),
            daemon=False,
        )
        t11 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client11),
            daemon=False,
        )
        t12 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client12),
            daemon=False,
        )
        t13 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client13),
            daemon=False,
        )
        t14 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client14),
            daemon=False,
        )
        t15 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client15),
            daemon=False,
        )
        t16 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client16),
            daemon=False,
        )
        t17 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client17),
            daemon=False,
        )
        t18 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client18),
            daemon=False,
        )
        t19 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client19),
            daemon=False,
        )
        t20 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client20),
            daemon=False,
        )
        t21 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client21),
            daemon=False,
        )
        t22 = threading.Thread(
            target=download1,
            args=(start, end, interval, q, dir, "1m", client22),
            daemon=False,
        )

        thrds = [
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t8,
            t9,
            t10,
            t11,
            t12,
            t13,
            t14,
            t15,
            t16,
            t17,
            t18,
            t19,
            t20,
            t21,
            t22,
        ]

        # thrds = [t1,t2,t3,t4,t5,t6,t7,t8]
        if q.qsize() < 22:
            for en, thrd in enumerate(range(q.qsize())):
                thrds[en].start()
            for en, thrd in enumerate(range(q.qsize())):
                thrds[en].join()
        else:
            t1.start()
            # time.sleep(5)
            t2.start()
            # time.sleep(5)
            t3.start()
            # time.sleep(5)
            t4.start()
            # time.sleep(5)
            t5.start()
            # time.sleep(5)
            t6.start()
            # time.sleep(5)
            t7.start()
            # time.sleep(5)
            t8.start()
            # time.sleep(5)
            t9.start()
            # time.sleep(5)
            t10.start()
            # time.sleep(5)
            t11.start()
            # time.sleep(5)
            t12.start()
            # time.sleep(5)
            t13.start()
            # time.sleep(5)
            t14.start()
            # time.sleep(5)
            t15.start()
            # time.sleep(5)
            t16.start()
            # time.sleep(5)
            t17.start()
            # time.sleep(5)
            t18.start()
            # time.sleep(5)
            t19.start()
            # time.sleep(5)
            t20.start()
            # time.sleep(5)
            t21.start()
            # time.sleep(5)
            t22.start()
            # time.sleep(5)

            t1.join()
            t2.join()
            t3.join()
            t4.join()
            t5.join()
            t6.join()
            t7.join()
            t8.join()
            t9.join()
            t10.join()
            t11.join()
            t12.join()
            t13.join()
            t14.join()
            t15.join()
            t16.join()
            t17.join()
            t18.join()
            t19.join()
            t20.join()
            t21.join()
            t22.join()
            # idc += 1
            # if idc == 3:
            #     idc = 0
            #     time.sleep(1)


def delete_incompletes(pth):
    def import_csv(csvfilename):
        data = []
        row_index = 0
        with open(csvfilename, "r", encoding="utf-8", errors="ignore") as scraped:
            reader = csv.reader(scraped, delimiter=",")
            for row in reader:
                if row:  # avoid blank lines
                    row_index += 1
                    columns = [str(row_index), row[0]]
                    data.append(columns)
        return data

    for a in os.listdir(pth):
        data = import_csv(os.path.join(pth, a))
        last_row = data[-1]
        if last_row[1] != "2022-01-04 23:59:00":
            # print(a, ":", last_row[1])
            os.remove(os.path.join(pth, a))


def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds
    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str
    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60, "w": 7 * 24 * 60 * 60}

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms


def clean_assets(client):
    dustables = client.get_dust_assets()
    dust_list = ""
    for en, item in enumerate(dustables["details"]):
        if float(item["toBNB"]) == 0:
            continue

        if item["asset"] == "BUSD":
            continue
        if item["asset"] == "VTHO":
            continue
        if item["asset"] == "VTHOBUSD":
            continue
        if (en + 1) < len(dustables["details"]):
            dust_list = dust_list + item["asset"] + ","
        else:
            dust_list = dust_list + item["asset"]
    dst = client.transfer_dust(asset=dust_list)
    # print(dst)


def get_acct(client):
    snaps = client.get_account()
    snap = []
    for ast in snaps["balances"]:
        if not float(ast["free"]) == 0:
            snap.append(ast)

    return snap


def run5m(d):
    global client1
    global client2
    global client3
    global client4
    global client5
    global client6
    global client7
    global client8
    global client9
    global client10
    global client11
    global client12
    global client13
    global client14
    global client15
    global client16
    global client17
    global client18
    global client19
    global client20
    global client21
    global client22
    client1 = Client(
        "igEARWI7LNtjhzHa3zrNAMtLlLtUjnNb3VFHSHCf5Nlnga4h3vAzthAQKe8wLYlC",
        "BM8EVK6TI5kHKQ7sORXpkwHet8mtq8alhOV5JJQ25kAIunKL7YkGgfc80inJad0I",
    )
    client2 = Client(
        "FCbtPk3mQj2IqpFbvR5rgPdXgZL8O3s4634zP5thOb0ob6MuiG7sxsvdzVy3MSe2",
        "aljR5fx3pHWSEc6JRkN0YMNlNk28rdM5CBE1XCzUvi8MQy4qoG5q7T5QAD3V9E1w",
    )
    client3 = Client(
        "yNt4nLpNc4sg4l7ZwFf3uqBbRq2YidMzIrrmAqWNjQMGCcPvTt66CXMl4S7LGyqO",
        "EWx8Fh5VjQrE6PGA9ywkIiAuOs0VX9Hk22dsBgLFdV4EqkE765ov5GxFDCpdbbr0",
    )
    client4 = Client(
        "zS74mTu25foQTQP2ttuH6gRN2cfbSnBzuIDMlktZRVvCQvHpq4G3CvQeHEszlSjH",
        "MFxQEWT28n7UQZ3JK6XsXcsWsRp0l8dYPvrZbQOxN9jK5ez11gqnXDt4aCaRJUFO",
    )
    client5 = Client(
        "xVHPQDvR2mvDITO9pRi2yFxhxmv1AyqS8cCxJepbH74Kt6XeB9Zn5lTmgbauSg1d",
        "sGx5MBrbi2iMpZKTvwurEuHA1OYPqSwT9DeJjC8ppkmWSokTDjZsbRjmq58nBG1c",
    )
    client6 = Client(
        "J9HNtB1mXWiqwaOLyxOQTB6yiy6Vg7ZfLOXdTdYPofc2hI8XDBcuc7yeIv02EtUx",
        "P1aQFvKuFyGOFgnufWKr61o0lPWuQjt1ZwzDDzZ1RsxMuUFDUiqx5uqI4JlT6sPJ",
    )
    client7 = Client(
        "ZQCSfbLRqUQifBmltgLf30Lm9gHSiRovZVyAhvsxi7nKA3TEC9ehsnsl1sdkqSct",
        "iIZOWSTGfTsdA1krWEz2sGU0pRjyGVzFUinMkq6eGwgIf45dorp6xCuxIHhXFQdt",
    )
    client8 = Client(
        "DocIMuZHP2x0TxprspUUX0eJSDwXsDkhKLvSa0TF6bKH69otCpSLXYqnKPLQsdzv",
        "j8KqFMMET2QzobtX82pe7VhjuA2eWz9ucc088RHgzKLD16QpHFUAVm4QZ4naIDWr",
    )

    client9 = Client(
        "DRxgIOPKpRZXSIhYHafd5CMdVIapwuaoG4AIjO8mzBYD7bTQ06pPXFXOf76ECwOE",
        "PYimvftBK6hyHGIPIfhIcEXW6vuCPwnpxqFPTWsUNkH8ZWbUzlV2oiZRbFrsl1uz",
    )
    client10 = Client(
        "BrJ0L1b0cja76YUA5QRbzKaGtAvBafSjRSC7dsB81kFhozFAr54rACI19YQlrmwA",
        "7nmhMtEqTFhNqNa2dFsdqnfYxg6V1YaDey6Ar1VPzuT6KwHV3BwLurQLiUJaUrnc",
    )
    client11 = Client(
        "luJhgk8T9lLUXnOdZEz7479yEscXNyAjV73cPUVrMVXMBUfmQUZFpHfXLmuXSsBo",
        "xmwL8GQQYurkb6mm9sFeu7hmLVXhnsRqDAvSffW9ueKYnkX7vayRVRN4BUL7b2eb",
    )
    client12 = Client(
        "EuSQ2E9ACuWhjk8vzchhwKQF6jCekw5Frfz8oyhvUdiybofGQ8m6BWZQj4d9hGRR",
        "3DNR0DYGmc88l5Q3b05bv4cmN5c3PdkjKU5niWRIYAEIHxCztKXGYoILdgeWuPzj",
    )
    client13 = Client(
        "TlpUkKHxa954Etozb3W2sW5aBL6IQPKksnWsOBj862jahC5rZ5euoYYlMVUaFhsp",
        "r8UZpFYHHHdB7TIqethm1Tr9HwhwHFn419C0vwoIDDICLtzghmWUYwGXfoNHlmUL",
    )
    client14 = Client(
        "CUG1uyFFLoCWjZbXqGXdyaIPxMQnC0SjmRi9LUDbx7OM2ggpoajXz7E6wbSDuKz1",
        "w7DHdcBWYN7o7h7GpjbFSEThocBp0lPDAAtujI1pMPwRmQk6mpoFva95xWI57JrG",
    )
    client15 = Client(
        "dUJAl6AkovWxuiYDRdieea2sY9nIn7w0XvhyDcJGd1YwwaKw7r1wnArJ9cypnaSm",
        "s5lQrvZzadltq02TphaSCFrQJYqekwrZq2UXwDmSudT7stScuPiERQDQ3awoYK40",
    )
    client16 = Client(
        "8wEz55YPNFSX5cmKJQ02CuAaYDmj3XkTcKlcZTG4dqEMIUmNGaEAhK4IG1I9ZAZD",
        "xKl9hrvoKTNALmto3SD0Aml2rxazluMxetzgLQxDDCxRtM0JIvIvHZQXEkWUdwKc",
    )
    client17 = Client(
        "569iIuo4OfFvQeRcdJp6N9qdaUVeETH1VGqxZtcuBVg5QD0miL2wZvC1aujuCusc",
        "VA4FRfERjp4Ia1TkfcsQG53w32n2M2pRkCMKboLAFQP3DuXrAP1pWJuO7zNKFtNz",
    )
    client18 = Client(
        "vsFeRfLI5WhvfuglcyeH3GNSRpWXfOaUG8jyBqr2vtdcbgpzcyYtVJAgDq6ZzHxW",
        "OeuGHyMY7Z5eRryofyy7cfxFXHJuWaSX153hyzIGonfLKOBcOiYWByZ5MFfDyfdO",
    )
    client19 = Client(
        "PmFssmBV38mduXGM0XncgtIp4zRxqYvFshRxldPQPlipl1gUh0DT2NRNJLC7Yu7Q",
        "NTBTqLuiXkBVvs58efwvCGiuh7QSZQuBYk7IKauzGmzW2oG5jeO3uUcnmcs6j73k",
    )
    client20 = Client(
        "jGQT6qTOezhPSBRQXmjM8jCNaqx8Z6XRZaaQyxcBVKEpC3rBNazUedp6bnNPf5Jc",
        "sIfQPeHx7t3CfPjRP2v3etHwgKNlD5tDdVkIENvQxyIBmwm4a2LY5HP62DG21kRj",
    )
    client21 = Client(
        "g2i2QIP9t6ZoavqQN580bmhJ8f3m0hukNJIT1xtIXoD12BFTLHYjeCsfqfepzBLU",
        "LkepTZstmlH4LbdysQ95dwscDpBaP7sif3VeRhVh2OHqv88XCzH4NbPpcefUiuhl",
    )
    client22 = Client(
        "AmnrxedCx5sLrvVJKDMaOt0l6JGC9pTtAubBdRbqtfRU4qgecPvNDs2JzGPUdz8D",
        "RQ36d5hlNx32sXEixUddDnli7ACpdST45GY9WJfbJbk0cbVRqSpcPn57M25Ayuyn",
    )

    clilist = [
        client1,
        client2,
        client3,
        client4,
        client5,
        client6,
        client7,
        client8,
        client9,
        client10,
        client11,
        client12,
        client13,
        client14,
        client15,
        client16,
        client17,
        client18,
        client19,
        client20,
        client21,
        client22,
    ]
    csv_path = "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/5m-unprocessed-testing"
    qlib_dir = "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/5m-qlib-testing"
    if os.path.exists(csv_path):
        shutil.rmtree(csv_path)
    os.mkdir(csv_path)

    dmin = d
    # strt = dmin - timedelta(minutes=(5*121))
    # strt = dmin - timedelta(days=7)
    strt = dateparser.parse("2022-07-01 00:00:00")
    startdownload_5m(
        start=strt.strftime("%d %B, %Y, %H:%M:%S"),
        end=dmin.strftime("%d %B, %Y, %H:%M:%S"),
        dir=csv_path,
    )
    # #startdownload_5m(start="4 Jan, 2019", end=dmin.strftime("%d %B, %Y, %H:%M:%S"),dir=csv_path)
    # #time.sleep(30)
    checker.degunk("5m", d.strftime("%d %B, %Y, %H:%M:%S"))
    if os.path.exists(qlib_dir):
        shutil.rmtree(qlib_dir)
    os.mkdir(qlib_dir)
    for clin in clilist:
        clin.close_connection()
    b = dump_bin.DumpDataAll(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        include_fields="open,high,low,close,volume,factor",
        freq="5min",
    )
    b.dump()


def append5m(d):
    # global client1
    # global client2
    # global client3
    # global client4
    # global client5
    # global client6
    # global client7
    # global client8
    global client9
    global client10
    global client11
    global client12
    global client13
    global client14
    global client15
    global client16
    global client17
    global client18
    global client19
    global client20
    global client21
    global client22
    # client1 = Client('igEARWI7LNtjhzHa3zrNAMtLlLtUjnNb3VFHSHCf5Nlnga4h3vAzthAQKe8wLYlC', 'BM8EVK6TI5kHKQ7sORXpkwHet8mtq8alhOV5JJQ25kAIunKL7YkGgfc80inJad0I')
    # client2 = Client('FCbtPk3mQj2IqpFbvR5rgPdXgZL8O3s4634zP5thOb0ob6MuiG7sxsvdzVy3MSe2','aljR5fx3pHWSEc6JRkN0YMNlNk28rdM5CBE1XCzUvi8MQy4qoG5q7T5QAD3V9E1w')
    # client3 = Client('yNt4nLpNc4sg4l7ZwFf3uqBbRq2YidMzIrrmAqWNjQMGCcPvTt66CXMl4S7LGyqO','EWx8Fh5VjQrE6PGA9ywkIiAuOs0VX9Hk22dsBgLFdV4EqkE765ov5GxFDCpdbbr0')
    # client4 = Client('zS74mTu25foQTQP2ttuH6gRN2cfbSnBzuIDMlktZRVvCQvHpq4G3CvQeHEszlSjH','MFxQEWT28n7UQZ3JK6XsXcsWsRp0l8dYPvrZbQOxN9jK5ez11gqnXDt4aCaRJUFO')
    # client5 = Client('xVHPQDvR2mvDITO9pRi2yFxhxmv1AyqS8cCxJepbH74Kt6XeB9Zn5lTmgbauSg1d','sGx5MBrbi2iMpZKTvwurEuHA1OYPqSwT9DeJjC8ppkmWSokTDjZsbRjmq58nBG1c')
    # client6 = Client('J9HNtB1mXWiqwaOLyxOQTB6yiy6Vg7ZfLOXdTdYPofc2hI8XDBcuc7yeIv02EtUx','P1aQFvKuFyGOFgnufWKr61o0lPWuQjt1ZwzDDzZ1RsxMuUFDUiqx5uqI4JlT6sPJ')
    # client7 = Client('ZQCSfbLRqUQifBmltgLf30Lm9gHSiRovZVyAhvsxi7nKA3TEC9ehsnsl1sdkqSct','iIZOWSTGfTsdA1krWEz2sGU0pRjyGVzFUinMkq6eGwgIf45dorp6xCuxIHhXFQdt')
    # client8 = Client('DocIMuZHP2x0TxprspUUX0eJSDwXsDkhKLvSa0TF6bKH69otCpSLXYqnKPLQsdzv','j8KqFMMET2QzobtX82pe7VhjuA2eWz9ucc088RHgzKLD16QpHFUAVm4QZ4naIDWr')

    client9 = Client(
        "DRxgIOPKpRZXSIhYHafd5CMdVIapwuaoG4AIjO8mzBYD7bTQ06pPXFXOf76ECwOE",
        "PYimvftBK6hyHGIPIfhIcEXW6vuCPwnpxqFPTWsUNkH8ZWbUzlV2oiZRbFrsl1uz",
    )
    client10 = Client(
        "BrJ0L1b0cja76YUA5QRbzKaGtAvBafSjRSC7dsB81kFhozFAr54rACI19YQlrmwA",
        "7nmhMtEqTFhNqNa2dFsdqnfYxg6V1YaDey6Ar1VPzuT6KwHV3BwLurQLiUJaUrnc",
    )
    client11 = Client(
        "luJhgk8T9lLUXnOdZEz7479yEscXNyAjV73cPUVrMVXMBUfmQUZFpHfXLmuXSsBo",
        "xmwL8GQQYurkb6mm9sFeu7hmLVXhnsRqDAvSffW9ueKYnkX7vayRVRN4BUL7b2eb",
    )
    client12 = Client(
        "EuSQ2E9ACuWhjk8vzchhwKQF6jCekw5Frfz8oyhvUdiybofGQ8m6BWZQj4d9hGRR",
        "3DNR0DYGmc88l5Q3b05bv4cmN5c3PdkjKU5niWRIYAEIHxCztKXGYoILdgeWuPzj",
    )
    client13 = Client(
        "TlpUkKHxa954Etozb3W2sW5aBL6IQPKksnWsOBj862jahC5rZ5euoYYlMVUaFhsp",
        "r8UZpFYHHHdB7TIqethm1Tr9HwhwHFn419C0vwoIDDICLtzghmWUYwGXfoNHlmUL",
    )
    client14 = Client(
        "CUG1uyFFLoCWjZbXqGXdyaIPxMQnC0SjmRi9LUDbx7OM2ggpoajXz7E6wbSDuKz1",
        "w7DHdcBWYN7o7h7GpjbFSEThocBp0lPDAAtujI1pMPwRmQk6mpoFva95xWI57JrG",
    )
    client15 = Client(
        "dUJAl6AkovWxuiYDRdieea2sY9nIn7w0XvhyDcJGd1YwwaKw7r1wnArJ9cypnaSm",
        "s5lQrvZzadltq02TphaSCFrQJYqekwrZq2UXwDmSudT7stScuPiERQDQ3awoYK40",
    )
    client16 = Client(
        "8wEz55YPNFSX5cmKJQ02CuAaYDmj3XkTcKlcZTG4dqEMIUmNGaEAhK4IG1I9ZAZD",
        "xKl9hrvoKTNALmto3SD0Aml2rxazluMxetzgLQxDDCxRtM0JIvIvHZQXEkWUdwKc",
    )
    client17 = Client(
        "569iIuo4OfFvQeRcdJp6N9qdaUVeETH1VGqxZtcuBVg5QD0miL2wZvC1aujuCusc",
        "VA4FRfERjp4Ia1TkfcsQG53w32n2M2pRkCMKboLAFQP3DuXrAP1pWJuO7zNKFtNz",
    )
    client18 = Client(
        "vsFeRfLI5WhvfuglcyeH3GNSRpWXfOaUG8jyBqr2vtdcbgpzcyYtVJAgDq6ZzHxW",
        "OeuGHyMY7Z5eRryofyy7cfxFXHJuWaSX153hyzIGonfLKOBcOiYWByZ5MFfDyfdO",
    )
    client19 = Client(
        "PmFssmBV38mduXGM0XncgtIp4zRxqYvFshRxldPQPlipl1gUh0DT2NRNJLC7Yu7Q",
        "NTBTqLuiXkBVvs58efwvCGiuh7QSZQuBYk7IKauzGmzW2oG5jeO3uUcnmcs6j73k",
    )
    client20 = Client(
        "jGQT6qTOezhPSBRQXmjM8jCNaqx8Z6XRZaaQyxcBVKEpC3rBNazUedp6bnNPf5Jc",
        "sIfQPeHx7t3CfPjRP2v3etHwgKNlD5tDdVkIENvQxyIBmwm4a2LY5HP62DG21kRj",
    )
    client21 = Client(
        "g2i2QIP9t6ZoavqQN580bmhJ8f3m0hukNJIT1xtIXoD12BFTLHYjeCsfqfepzBLU",
        "LkepTZstmlH4LbdysQ95dwscDpBaP7sif3VeRhVh2OHqv88XCzH4NbPpcefUiuhl",
    )
    client22 = Client(
        "AmnrxedCx5sLrvVJKDMaOt0l6JGC9pTtAubBdRbqtfRU4qgecPvNDs2JzGPUdz8D",
        "RQ36d5hlNx32sXEixUddDnli7ACpdST45GY9WJfbJbk0cbVRqSpcPn57M25Ayuyn",
    )

    clilist = [
        # client1,
        # client2,
        # client3,
        # client4,
        # client5,
        # client6,
        # client7,
        # client8,
        client9,
        client10,
        client11,
        client12,
        client13,
        client14,
        client15,
        client16,
        client17,
        client18,
        client19,
        client20,
        client21,
        client22,
    ]
    csv_path = (
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/5m-temp"
    )
    qlib_dir = "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/5m-qlib-temp"
    if os.path.exists(csv_path):
        shutil.rmtree(csv_path)
    os.mkdir(csv_path)
    strd = d.strftime("%d %B, %Y, %H:%M:%S")
    dmin = d + timedelta(minutes=60)
    dmax = d - timedelta(days=1)
    with open(
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/5m-qlib-temp/calendars/5min.txt"
    ) as f:
        read = csv.reader(f)
        cl = []
        for line in read:
            cl.append(line)
        last = cl[-1][0]
    lastd = dateparser.parse(last)
    # lastd = lastd - timedelta(days=1)
    lastd = lastd - timedelta(minutes=5)

    startdownload_5m(
        start=lastd.strftime("%d %B, %Y, %H:%M:%S"),
        end=dmin.strftime("%d %B, %Y, %H:%M:%S"),
        dir=csv_path,
        app=True,
    )
    time.sleep(2)
    # checker.degunkapp('5m',strd)
    # if os.path.exists(qlib_dir):
    #     shutil.rmtree(qlib_dir)
    # os.mkdir(qlib_dir)
    for clin in clilist:
        clin.close_connection()
    if not os.listdir(csv_path) == []:
        b = dump_bin.DumpDataUpdate(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            include_fields="open,high,low,close,volume,factor",
            freq="5min",
            max_workers=4,
        )
        b.dump()


def run1m(d):
    global client1
    global client2
    global client3
    global client4
    global client5
    global client6
    global client7
    global client8
    global client9
    global client10
    global client11
    global client12
    global client13
    global client14
    global client15
    global client16
    global client17
    global client18
    global client19
    global client20
    global client21
    global client22
    client1 = Client(
        "igEARWI7LNtjhzHa3zrNAMtLlLtUjnNb3VFHSHCf5Nlnga4h3vAzthAQKe8wLYlC",
        "BM8EVK6TI5kHKQ7sORXpkwHet8mtq8alhOV5JJQ25kAIunKL7YkGgfc80inJad0I",
    )
    client2 = Client(
        "FCbtPk3mQj2IqpFbvR5rgPdXgZL8O3s4634zP5thOb0ob6MuiG7sxsvdzVy3MSe2",
        "aljR5fx3pHWSEc6JRkN0YMNlNk28rdM5CBE1XCzUvi8MQy4qoG5q7T5QAD3V9E1w",
    )
    client3 = Client(
        "yNt4nLpNc4sg4l7ZwFf3uqBbRq2YidMzIrrmAqWNjQMGCcPvTt66CXMl4S7LGyqO",
        "EWx8Fh5VjQrE6PGA9ywkIiAuOs0VX9Hk22dsBgLFdV4EqkE765ov5GxFDCpdbbr0",
    )
    client4 = Client(
        "zS74mTu25foQTQP2ttuH6gRN2cfbSnBzuIDMlktZRVvCQvHpq4G3CvQeHEszlSjH",
        "MFxQEWT28n7UQZ3JK6XsXcsWsRp0l8dYPvrZbQOxN9jK5ez11gqnXDt4aCaRJUFO",
    )
    client5 = Client(
        "xVHPQDvR2mvDITO9pRi2yFxhxmv1AyqS8cCxJepbH74Kt6XeB9Zn5lTmgbauSg1d",
        "sGx5MBrbi2iMpZKTvwurEuHA1OYPqSwT9DeJjC8ppkmWSokTDjZsbRjmq58nBG1c",
    )
    client6 = Client(
        "J9HNtB1mXWiqwaOLyxOQTB6yiy6Vg7ZfLOXdTdYPofc2hI8XDBcuc7yeIv02EtUx",
        "P1aQFvKuFyGOFgnufWKr61o0lPWuQjt1ZwzDDzZ1RsxMuUFDUiqx5uqI4JlT6sPJ",
    )
    client7 = Client(
        "ZQCSfbLRqUQifBmltgLf30Lm9gHSiRovZVyAhvsxi7nKA3TEC9ehsnsl1sdkqSct",
        "iIZOWSTGfTsdA1krWEz2sGU0pRjyGVzFUinMkq6eGwgIf45dorp6xCuxIHhXFQdt",
    )
    client8 = Client(
        "DocIMuZHP2x0TxprspUUX0eJSDwXsDkhKLvSa0TF6bKH69otCpSLXYqnKPLQsdzv",
        "j8KqFMMET2QzobtX82pe7VhjuA2eWz9ucc088RHgzKLD16QpHFUAVm4QZ4naIDWr",
    )

    client9 = Client(
        "DRxgIOPKpRZXSIhYHafd5CMdVIapwuaoG4AIjO8mzBYD7bTQ06pPXFXOf76ECwOE",
        "PYimvftBK6hyHGIPIfhIcEXW6vuCPwnpxqFPTWsUNkH8ZWbUzlV2oiZRbFrsl1uz",
    )
    client10 = Client(
        "BrJ0L1b0cja76YUA5QRbzKaGtAvBafSjRSC7dsB81kFhozFAr54rACI19YQlrmwA",
        "7nmhMtEqTFhNqNa2dFsdqnfYxg6V1YaDey6Ar1VPzuT6KwHV3BwLurQLiUJaUrnc",
    )
    client11 = Client(
        "luJhgk8T9lLUXnOdZEz7479yEscXNyAjV73cPUVrMVXMBUfmQUZFpHfXLmuXSsBo",
        "xmwL8GQQYurkb6mm9sFeu7hmLVXhnsRqDAvSffW9ueKYnkX7vayRVRN4BUL7b2eb",
    )
    client12 = Client(
        "EuSQ2E9ACuWhjk8vzchhwKQF6jCekw5Frfz8oyhvUdiybofGQ8m6BWZQj4d9hGRR",
        "3DNR0DYGmc88l5Q3b05bv4cmN5c3PdkjKU5niWRIYAEIHxCztKXGYoILdgeWuPzj",
    )
    client13 = Client(
        "TlpUkKHxa954Etozb3W2sW5aBL6IQPKksnWsOBj862jahC5rZ5euoYYlMVUaFhsp",
        "r8UZpFYHHHdB7TIqethm1Tr9HwhwHFn419C0vwoIDDICLtzghmWUYwGXfoNHlmUL",
    )
    client14 = Client(
        "CUG1uyFFLoCWjZbXqGXdyaIPxMQnC0SjmRi9LUDbx7OM2ggpoajXz7E6wbSDuKz1",
        "w7DHdcBWYN7o7h7GpjbFSEThocBp0lPDAAtujI1pMPwRmQk6mpoFva95xWI57JrG",
    )
    client15 = Client(
        "dUJAl6AkovWxuiYDRdieea2sY9nIn7w0XvhyDcJGd1YwwaKw7r1wnArJ9cypnaSm",
        "s5lQrvZzadltq02TphaSCFrQJYqekwrZq2UXwDmSudT7stScuPiERQDQ3awoYK40",
    )
    client16 = Client(
        "8wEz55YPNFSX5cmKJQ02CuAaYDmj3XkTcKlcZTG4dqEMIUmNGaEAhK4IG1I9ZAZD",
        "xKl9hrvoKTNALmto3SD0Aml2rxazluMxetzgLQxDDCxRtM0JIvIvHZQXEkWUdwKc",
    )
    client17 = Client(
        "569iIuo4OfFvQeRcdJp6N9qdaUVeETH1VGqxZtcuBVg5QD0miL2wZvC1aujuCusc",
        "VA4FRfERjp4Ia1TkfcsQG53w32n2M2pRkCMKboLAFQP3DuXrAP1pWJuO7zNKFtNz",
    )
    client18 = Client(
        "vsFeRfLI5WhvfuglcyeH3GNSRpWXfOaUG8jyBqr2vtdcbgpzcyYtVJAgDq6ZzHxW",
        "OeuGHyMY7Z5eRryofyy7cfxFXHJuWaSX153hyzIGonfLKOBcOiYWByZ5MFfDyfdO",
    )
    client19 = Client(
        "PmFssmBV38mduXGM0XncgtIp4zRxqYvFshRxldPQPlipl1gUh0DT2NRNJLC7Yu7Q",
        "NTBTqLuiXkBVvs58efwvCGiuh7QSZQuBYk7IKauzGmzW2oG5jeO3uUcnmcs6j73k",
    )
    client20 = Client(
        "jGQT6qTOezhPSBRQXmjM8jCNaqx8Z6XRZaaQyxcBVKEpC3rBNazUedp6bnNPf5Jc",
        "sIfQPeHx7t3CfPjRP2v3etHwgKNlD5tDdVkIENvQxyIBmwm4a2LY5HP62DG21kRj",
    )
    client21 = Client(
        "g2i2QIP9t6ZoavqQN580bmhJ8f3m0hukNJIT1xtIXoD12BFTLHYjeCsfqfepzBLU",
        "LkepTZstmlH4LbdysQ95dwscDpBaP7sif3VeRhVh2OHqv88XCzH4NbPpcefUiuhl",
    )
    client22 = Client(
        "AmnrxedCx5sLrvVJKDMaOt0l6JGC9pTtAubBdRbqtfRU4qgecPvNDs2JzGPUdz8D",
        "RQ36d5hlNx32sXEixUddDnli7ACpdST45GY9WJfbJbk0cbVRqSpcPn57M25Ayuyn",
    )

    clilist = [
        client1,
        client2,
        client3,
        client4,
        client5,
        client6,
        client7,
        client8,
        client9,
        client10,
        client11,
        client12,
        client13,
        client14,
        client15,
        client16,
        client17,
        client18,
        client19,
        client20,
        client21,
        client22,
    ]
    csv_path = "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-unprocessed-t"
    qlib_dir = (
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib-t"
    )
    # if os.path.exists(csv_path):
    #     shutil.rmtree(csv_path)
    # os.mkdir(csv_path)

    dmin = d

    strt = dateparser.parse("2019-01-01 00:00:00")
    # startdownload_1m(
    #     start=strt.strftime("%d %B, %Y, %H:%M:%S"),
    #     end=dmin.strftime("%d %B, %Y, %H:%M:%S"),
    #     dir=csv_path,
    # )
    # #time.sleep(30)
    checker.degunk("1m", d.strftime("%d %B, %Y, %H:%M:%S"))
    if os.path.exists(qlib_dir):
        shutil.rmtree(qlib_dir)
    os.mkdir(qlib_dir)
    for clin in clilist:
        clin.close_connection()
    b = dump_bin.DumpDataAll(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        include_fields="open,high,low,close,volume,factor",
        freq="1min",
    )
    b.dump()


# @profile
def append1m(d):
    global client1
    global client2
    global client3
    global client4
    global client5
    global client6
    global client7
    global client8
    global client9
    global client10
    global client11
    global client12
    global client13
    global client14
    global client15
    global client16
    global client17
    global client18
    global client19
    global client20
    global client21
    global client22
    client1 = Client(
        "igEARWI7LNtjhzHa3zrNAMtLlLtUjnNb3VFHSHCf5Nlnga4h3vAzthAQKe8wLYlC",
        "BM8EVK6TI5kHKQ7sORXpkwHet8mtq8alhOV5JJQ25kAIunKL7YkGgfc80inJad0I",
    )
    client2 = Client(
        "FCbtPk3mQj2IqpFbvR5rgPdXgZL8O3s4634zP5thOb0ob6MuiG7sxsvdzVy3MSe2",
        "aljR5fx3pHWSEc6JRkN0YMNlNk28rdM5CBE1XCzUvi8MQy4qoG5q7T5QAD3V9E1w",
    )
    client3 = Client(
        "yNt4nLpNc4sg4l7ZwFf3uqBbRq2YidMzIrrmAqWNjQMGCcPvTt66CXMl4S7LGyqO",
        "EWx8Fh5VjQrE6PGA9ywkIiAuOs0VX9Hk22dsBgLFdV4EqkE765ov5GxFDCpdbbr0",
    )
    client4 = Client(
        "zS74mTu25foQTQP2ttuH6gRN2cfbSnBzuIDMlktZRVvCQvHpq4G3CvQeHEszlSjH",
        "MFxQEWT28n7UQZ3JK6XsXcsWsRp0l8dYPvrZbQOxN9jK5ez11gqnXDt4aCaRJUFO",
    )
    client5 = Client(
        "xVHPQDvR2mvDITO9pRi2yFxhxmv1AyqS8cCxJepbH74Kt6XeB9Zn5lTmgbauSg1d",
        "sGx5MBrbi2iMpZKTvwurEuHA1OYPqSwT9DeJjC8ppkmWSokTDjZsbRjmq58nBG1c",
    )
    client6 = Client(
        "J9HNtB1mXWiqwaOLyxOQTB6yiy6Vg7ZfLOXdTdYPofc2hI8XDBcuc7yeIv02EtUx",
        "P1aQFvKuFyGOFgnufWKr61o0lPWuQjt1ZwzDDzZ1RsxMuUFDUiqx5uqI4JlT6sPJ",
    )
    client7 = Client(
        "ZQCSfbLRqUQifBmltgLf30Lm9gHSiRovZVyAhvsxi7nKA3TEC9ehsnsl1sdkqSct",
        "iIZOWSTGfTsdA1krWEz2sGU0pRjyGVzFUinMkq6eGwgIf45dorp6xCuxIHhXFQdt",
    )
    client8 = Client(
        "DocIMuZHP2x0TxprspUUX0eJSDwXsDkhKLvSa0TF6bKH69otCpSLXYqnKPLQsdzv",
        "j8KqFMMET2QzobtX82pe7VhjuA2eWz9ucc088RHgzKLD16QpHFUAVm4QZ4naIDWr",
    )

    client9 = Client(
        "DRxgIOPKpRZXSIhYHafd5CMdVIapwuaoG4AIjO8mzBYD7bTQ06pPXFXOf76ECwOE",
        "PYimvftBK6hyHGIPIfhIcEXW6vuCPwnpxqFPTWsUNkH8ZWbUzlV2oiZRbFrsl1uz",
    )
    client10 = Client(
        "BrJ0L1b0cja76YUA5QRbzKaGtAvBafSjRSC7dsB81kFhozFAr54rACI19YQlrmwA",
        "7nmhMtEqTFhNqNa2dFsdqnfYxg6V1YaDey6Ar1VPzuT6KwHV3BwLurQLiUJaUrnc",
    )
    client11 = Client(
        "luJhgk8T9lLUXnOdZEz7479yEscXNyAjV73cPUVrMVXMBUfmQUZFpHfXLmuXSsBo",
        "xmwL8GQQYurkb6mm9sFeu7hmLVXhnsRqDAvSffW9ueKYnkX7vayRVRN4BUL7b2eb",
    )
    client12 = Client(
        "EuSQ2E9ACuWhjk8vzchhwKQF6jCekw5Frfz8oyhvUdiybofGQ8m6BWZQj4d9hGRR",
        "3DNR0DYGmc88l5Q3b05bv4cmN5c3PdkjKU5niWRIYAEIHxCztKXGYoILdgeWuPzj",
    )
    client13 = Client(
        "TlpUkKHxa954Etozb3W2sW5aBL6IQPKksnWsOBj862jahC5rZ5euoYYlMVUaFhsp",
        "r8UZpFYHHHdB7TIqethm1Tr9HwhwHFn419C0vwoIDDICLtzghmWUYwGXfoNHlmUL",
    )
    client14 = Client(
        "CUG1uyFFLoCWjZbXqGXdyaIPxMQnC0SjmRi9LUDbx7OM2ggpoajXz7E6wbSDuKz1",
        "w7DHdcBWYN7o7h7GpjbFSEThocBp0lPDAAtujI1pMPwRmQk6mpoFva95xWI57JrG",
    )
    client15 = Client(
        "dUJAl6AkovWxuiYDRdieea2sY9nIn7w0XvhyDcJGd1YwwaKw7r1wnArJ9cypnaSm",
        "s5lQrvZzadltq02TphaSCFrQJYqekwrZq2UXwDmSudT7stScuPiERQDQ3awoYK40",
    )
    client16 = Client(
        "8wEz55YPNFSX5cmKJQ02CuAaYDmj3XkTcKlcZTG4dqEMIUmNGaEAhK4IG1I9ZAZD",
        "xKl9hrvoKTNALmto3SD0Aml2rxazluMxetzgLQxDDCxRtM0JIvIvHZQXEkWUdwKc",
    )
    client17 = Client(
        "569iIuo4OfFvQeRcdJp6N9qdaUVeETH1VGqxZtcuBVg5QD0miL2wZvC1aujuCusc",
        "VA4FRfERjp4Ia1TkfcsQG53w32n2M2pRkCMKboLAFQP3DuXrAP1pWJuO7zNKFtNz",
    )
    client18 = Client(
        "vsFeRfLI5WhvfuglcyeH3GNSRpWXfOaUG8jyBqr2vtdcbgpzcyYtVJAgDq6ZzHxW",
        "OeuGHyMY7Z5eRryofyy7cfxFXHJuWaSX153hyzIGonfLKOBcOiYWByZ5MFfDyfdO",
    )
    client19 = Client(
        "PmFssmBV38mduXGM0XncgtIp4zRxqYvFshRxldPQPlipl1gUh0DT2NRNJLC7Yu7Q",
        "NTBTqLuiXkBVvs58efwvCGiuh7QSZQuBYk7IKauzGmzW2oG5jeO3uUcnmcs6j73k",
    )
    client20 = Client(
        "jGQT6qTOezhPSBRQXmjM8jCNaqx8Z6XRZaaQyxcBVKEpC3rBNazUedp6bnNPf5Jc",
        "sIfQPeHx7t3CfPjRP2v3etHwgKNlD5tDdVkIENvQxyIBmwm4a2LY5HP62DG21kRj",
    )
    client21 = Client(
        "g2i2QIP9t6ZoavqQN580bmhJ8f3m0hukNJIT1xtIXoD12BFTLHYjeCsfqfepzBLU",
        "LkepTZstmlH4LbdysQ95dwscDpBaP7sif3VeRhVh2OHqv88XCzH4NbPpcefUiuhl",
    )
    client22 = Client(
        "AmnrxedCx5sLrvVJKDMaOt0l6JGC9pTtAubBdRbqtfRU4qgecPvNDs2JzGPUdz8D",
        "RQ36d5hlNx32sXEixUddDnli7ACpdST45GY9WJfbJbk0cbVRqSpcPn57M25Ayuyn",
    )

    clilist = [
        client1,
        client2,
        client3,
        client4,
        client5,
        client6,
        client7,
        client8,
        client9,
        client10,
        client11,
        client12,
        client13,
        client14,
        client15,
        client16,
        client17,
        client18,
        client19,
        client20,
        client21,
        client22,
    ]
    csv_path = (
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-temp"
    )
    qlib_dir = (
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib"
    )
    if os.path.exists(csv_path):
        shutil.rmtree(csv_path)
    os.mkdir(csv_path)
    strd = d.strftime("%d %B, %Y, %H:%M:%S")
    dmin = d + timedelta(minutes=60)
    dmax = d - timedelta(days=1)
    with open(
        "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib/calendars/1min.txt"
    ) as f:
        read = csv.reader(f)
        cl = []
        for line in read:
            cl.append(line)
        last = cl[-1][0]
    lastd = dateparser.parse(last)
    # lastd = lastd - timedelta(days=1)
    lastd = lastd - timedelta(minutes=5)

    startdownload_1m(
        start=lastd.strftime("%d %B, %Y, %H:%M:%S"),
        end=dmin.strftime("%d %B, %Y, %H:%M:%S"),
        dir=csv_path,
        app=True,
    )
    time.sleep(1)
    # checker.degunkapp('5m',strd)
    # if os.path.exists(qlib_dir):
    #     shutil.rmtree(qlib_dir)
    # os.mkdir(qlib_dir)
    for clin in clilist:
        clin.close_connection()
    if not os.listdir(csv_path) == []:
        b = dump_bin.DumpDataUpdate(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            include_fields="open,high,low,close,volume,factor",
            freq="1min",
            max_workers=16,
        )
        b.dump()


if __name__ == "__main__":
    d = datetime.now(pytz.utc)
    run1m(d)
    print("data download and prep time elapsed: {}".format(datetime.now(pytz.utc) - d))
