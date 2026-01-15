# Utilities for making sense of information in paramfiles a.k.a. paramdicts

def get_noise_model_tags_to_noise_model_infos(d):
    """Get a dictionary that maps "noise model tags" to the information for that
    noise model. A "noise model tag" is any base-level item in the paramdict
    that starts with the string 'noise_model'. I.e., each "noise model tag" will
    start with 'noise_model'.

    Parameters
    ----------
    d : dict
        The parsed parameter dictionary.

    Returns
    -------
    dict[str] -> dict
        The noise model tag points to another dictionary containing info for the
        noise model itself.
    """
    modeltags2modelinfos = {}
    for k, v in d.items():
        if k.startswith('noise_model'):
            modeltags2modelinfos[k] = v
    return modeltags2modelinfos
    
def get_mapnames_to_noise_model_tags(d=None, mapnames2minfos=None,
                                     modeltags2modelinfos=None):
    """Get a dictionary that maps "mapnames" to the "noise model tag" to which
    that mapname ultimately belongs. This is inferred by looping over all
    mapnames and collecting the 'qid' and possible 'subproduct_kwargs' out of 
    that mapname's 'noise_info' (this mapnames2noiseinfo is either provided
    implictly via a paramdict, or explictly as part of a mapnames2minfos dict, 
    which has keys of mapnames and values of dictionaries, each of which has a 
    'noise_info' key, e.g. mapname2noiseinfo['dr6_pa5_f090']['noise_info'].) 
    Then, the 'qid' and 'subproduct_kwargs' are matched against those of each
    possible noise model info. Exactly one match must be found.

    Parameters
    ----------
    d : dict, optional
        The parsed parameter dictionary. Used to get mapnames, mapname2noiseinfo,
        and modeltags2modelinfos. Takes priority over mapnames2minfos and
        modeltags2modelinfos if provided.
    mapnames2minfos : dict, optional
        A dictionary mapping mapnames to noise info via
        mapname2noiseinfo[mapname]['noise_info'], by default None. Must be 
        provided if d is None.
    modeltags2modelinfos : dict, optional
        A dictionary mapping noise model tags to noise model info, by default
        None. Must be provided if d is None.

    Returns
    -------
    dict[str] -> str
        A dictionary mapping mapnames to the tag of the noise model to which
        each map belongs.
    """
    if d is None:
        mapnames = mapnames2minfos.keys()
        mapname2noiseinfo = lambda x: mapnames2minfos[x]['noise_info']
    else: 
        mapnames = [f'{sv}_{m}' for sv in d['surveys'] for m in d[f'arrays_{sv}']]
        mapname2noiseinfo = lambda x: d[f'noise_info_{x}']
        modeltags2modelinfos = get_noise_model_tags_to_noise_model_infos(d)

    mapnames2modeltags = {}
    for mapname in mapnames:
        noise_info = mapname2noiseinfo(mapname)
        qid = noise_info['qid']
        subproduct_kwargs = noise_info.get('subproduct_kwargs', {}) # might be empty

        for tag, noise_model_info in modeltags2modelinfos.items():
            _qids = noise_model_info['qids']
            _subproduct_kwargs = noise_model_info.get('subproduct_kwargs', {})

            # match criteria
            qid_match = False
            subproduct_kwargs_match = False
            if qid in _qids:
                qid_match = True

                if len(subproduct_kwargs) > 0:
                    for k, v in subproduct_kwargs.items(): # 'inout_split', 'inout1'
                        if k in _subproduct_kwargs: # 'inout_split' in {'inout_split': ['inout1', 'inout2']}
                            if v in _subproduct_kwargs[k]: # 'inout1' in ['inout1', 'inout2']
                                subproduct_kwargs_match = True
                else:
                    if subproduct_kwargs == _subproduct_kwargs: # {} == {}
                        subproduct_kwargs_match = True
                    
            if qid_match and subproduct_kwargs_match:
                assert mapname not in mapnames2modeltags, \
                    f'{mapname=} matched to {tag} but already maps to noise ' + \
                    f'model {mapnames2modeltags[mapname]}'
                mapnames2modeltags[mapname] = tag

        assert mapname in mapnames2modeltags, \
            f'{mapname=} with {qid=} and {subproduct_kwargs=} did not match ' + \
            'with any noise_model_info'
                        
    return mapnames2modeltags