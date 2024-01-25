import asyncio
import aiohttp
from functools import reduce
from typing import Dict
from typing import List
import numpy as np
import pandas as pd
import json

import urllib
import unicodedata
from typing import List

greek_alphabet = 'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'

def encode_identifier(identifier: str) -> str:
    '''
    It prepares an identifier so that it can be passed to PUG REST.
    The following operations are performed:
    - greek letters are converted to their full english names (in lower case)
    - URL encoding
    :param identifier: input identifier
    :return: converted identifier that can be passed to PUG REST
    '''
    converted_identifier = ''
    for s in identifier:
        if s in greek_alphabet:
            converted_identifier += unicodedata.name(s).split()[-1].lower()
        else:
            converted_identifier += s
    converted_identifier = urllib.parse.quote(converted_identifier)
    return converted_identifier

def parse_TOCHeadings(sections: List, TOCHeading_path='', TOCHeading_paths = None):
    '''
    Returns the TOCHeadings in the index or full dataset returned by PUG view.
    THe function works recursively.

    :param sections: Pug view returns a dictionary in which the TOCHeadings are in the value ['Record']['Section']
    :param TOCHeading_path: not to be used (needed for recursion)
    :param TOCHeading_paths: not to be used (needed for recursion)
    :return:array of TOCHeadings in the index or full dataaset returned by PUG view
    '''
    if TOCHeading_paths is None:
        TOCHeading_paths = []
    for section in sections:
        tmp = TOCHeading_path + ('->' if TOCHeading_path else '') + section['TOCHeading']
        if 'Section' in section:
            parse_TOCHeadings(section['Section'], tmp, TOCHeading_paths)
        else:
            TOCHeading_paths.append(tmp)
    return TOCHeading_paths



async def retrieve_CID(identifier: str,
                       identifier_type: str,
                       session: aiohttp.ClientSession,
                       semaphore: asyncio.Semaphore,
                       timeout: float = 30.,
                       retries: int = 5) -> Dict:
    '''
    Retrieves the PubChem CID(s) from an identifier.
    For more information please see https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest-tutorial.
    :param identifier: the input identifier
    :param identifier_type: the input identifier type (so far we allow 'CAS number', 'EC number', or 'name')
    :param session: aiohttp client session
    :param semaphore: asyncio semaphore to control concurrency
    :param timeout: time out for the PubChem POST request (in sec)
    :param retries: number of retries of the REST request

    :return: dictionary with the CID(s)
    '''
    pubchem_base_url = r'https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
    url = pubchem_base_url + 'compound/name/cids/JSON'

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for _ in range(retries):
        try:
            # note the two context managers
            # https://stackoverflow.com/questions/66724841/using-a-semaphore-with-asyncio-in-python
            # we pass a dictionary that is updated by the client tracing
            trace_request_ctx = {'request duration': None}
            async with semaphore, session.post(url=url, timeout=timeout, trace_request_ctx=trace_request_ctx) as resp:

                original_response = await resp.json()

                # critical sleep to ensure that load does not exceed PubChem's thresholds
                min_time_per_request = 1.1
                if trace_request_ctx['request duration'] < min_time_per_request:
                    idle_time = min_time_per_request - trace_request_ctx['request duration']
                    await asyncio.sleep(idle_time)

                # successful response
                if resp.status == 200:
                    # retrieve the PubChem Compound IDs (CID) from the response json
                    CID_path = 'IdentifierList/CID'
                    CIDs = reduce(lambda x, p: x[p], CID_path.split('/'), original_response)
                    return CIDs
        except: pass


async def retrieve_pugview(CID: str,
                           session: aiohttp.ClientSession,
                           semaphore: asyncio.Semaphore,
                           type: str = 'data',
                           timeout: float = 30.,
                           retries: int = 5) -> Dict:
    '''
    Retrieves the index or the complete dataset of a compound using PubChem PUG View.
    For more information please see https://pubchemdocs.ncbi.nlm.nih.gov/pug-view.
    :param CID: The PubChem compound ID (CID)
    :param type: Specifies whether the index (type='index') or complete dataset (type='data') is returned
    :param session: aiohttp client session
    :param semaphore: asyncio semaphore to control concurrency
    :param timeout: time out for the PubChem POST request (in sec)
    :param retries: number of retries of the REST request
    :return: Dictionary with the index or complete dataset
    '''
    if type == 'info' or type == 'common_name':
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{CID}/JSON'
    elif type == 'data':
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CID}/JSON'
    elif type == 'synonyms':
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CID}/synonyms/JSON'
    # elif type == 'taxonomy':
    #     url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CID}/synonyms/JSON'
    else:
        raise ValueError("type must be 'index' or 'data'")



    for i_retry in range(retries):
        trace_request_ctx = {'request duration': 1.5}
        # try:
        async with semaphore, session.get(url=url, timeout=timeout, trace_request_ctx=trace_request_ctx) as resp:

            result = await resp.json()

            # Critical sleep to ensure that load does not exceed PubChem's thresholds
            min_time_per_request = 1.1
            if trace_request_ctx['request duration'] < min_time_per_request:
                idle_time = min_time_per_request - trace_request_ctx['request duration']
                await asyncio.sleep(idle_time)

            if resp.status == 200:  # Successful response
                print(f"Retrieved {CID}")
                return result
            elif resp.status == 503:  # PubChem server busy, we will retry
                if i_retry == retries - 1:
                    print(f"Retrying {CID}")
                    return result
            else:  # Unsuccessful response
                print(f"Failed {CID}")
                return np.nan
        # except:
        #     print(f"Error with {CID}")


async def get_info(identifiers: List[str],
                   identifier_type: str,
                   type: str = 'data'):

    # if identifier_type == 'name': data = 'name=' + encode_identifier(identifier)
    # else: data = 'name=' + identifier
    # data = data.encode(encoding='utf-8')

    async def on_request_start(session, trace_config_ctx, params):
        trace_config_ctx.start = asyncio.get_event_loop().time()

    async def on_request_end(session, trace_config_ctx, params):
        elapsed_time = asyncio.get_event_loop().time() - trace_config_ctx.start
        if trace_config_ctx.trace_request_ctx['request duration'] is not None:
            raise Exception('should not happen')
        trace_config_ctx.trace_request_ctx['request duration'] = elapsed_time

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_end.append(on_request_end)

    sem = asyncio.Semaphore(10000)
    session_timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(connector=None, timeout=session_timeout,
                                     trace_configs=[trace_config]) as session:
        # if identifier_type in ['name', 'CAS number', 'EC number']:
        #     # obtain the CID data
        #     tasks = []
        #     for identifier in identifiers:
        #         tasks.append(retrieve_CID(identifier, identifier_type,
        #                                 session=session, semaphore=sem))
        #     CID_results = await asyncio.gather(*tasks)
        #     CIDs = CID_results['CID'].explode().dropna().drop_duplicates().to_list()  # Obtain unique CIDs
        #     await asyncio.sleep(1.)
        # else:
        #     CIDs = identifiers

        CIDs = identifiers

        tasks = []
        for CID in CIDs:
            tasks.append(retrieve_pugview(str(CID), type=type, session=session, semaphore=sem))
        pugview_results = await asyncio.gather(*tasks)

    return pugview_results