def merge_objstats(target, source):
    for oid, data in source.items():
        sid = data["sid"]
        target.setdefault(oid, {})
        target[oid][sid] = data