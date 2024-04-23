

var finput, fList, contentDiv, dddisp, submitBtn, FileListLength = 0;
window.addEventListener('load', function(e) {
    finput = document.getElementById('formFile');
    fList = document.getElementById('uploadedFiles');
    contentDiv = document.getElementById('uploadContainer');
    dddisp = document.getElementById('dragDropDisplay');
    submitBtn = document.getElementById('submitBtn');
    finput.addEventListener('change', fileAddedHandler);
    contentDiv.ondragover = (event) => {
        event.preventDefault();
    }
    contentDiv.ondragenter = function(event) {
        event.preventDefault();
        _dragDropCount++;
        dddisp.classList.remove('d-none');
    }
    contentDiv.ondragleave = function(event) {
        _dragDropCount--;
        if (_dragDropCount <= 0)
            dddisp.classList.add('d-none');
    }
    contentDiv.ondrop = function(event) {
        console.log('d', event.dataTransfer.files);
        finput.files = event.dataTransfer.files;
        event.preventDefault();
        _dragDropCount = 0;
        dddisp.classList.add('d-none');
        fileAddedHandler();
    }
})
const _maxFileMegaBytes = 200;
var _dragDropCount = 0;

const svgid = {
    'image/jpeg' : 'filetype-jpg',
    'image/png' : 'filetype-png',
}
const mimetypes = {
    'image/jpeg' : 'JPG',
    'image/png' : 'PNG',
}

function getElementDisplay (ele) {
    let computedStyle = window.getComputedStyle(ele);
    // Get the value of the display property
    let displayValue = computedStyle.getPropertyValue("display");
    return displayValue;
}

function fListRowHTML(file, srctype, cnt) {
    let uid = uuidv4();
    let targettypes = getTargets(srctype);
    let options = targettypes.map(x => `<option value="${x}">${x}</option>`).join('');
    let content = new DocumentFragment();
    let parentr = document.createElement('tr');

    let filesize = "0B";
    if (file.size < 1024) {
        filesize = file.size.toFixed(2) + " B";
    } else if (file.size < 1024*1024) {
        filesize = (file.size/1024).toFixed(2) + " KB";
    } else {
        filesize = (file.size/(1024*1024)).toFixed(2) + " MB";
    }

    parentr.innerHTML =  `
    <input type="file" class="d-none" name="file_${uid}" form="submissionform"/>
    <input type="text" class="d-none" name="name_${uid}" value="${file.name}" form="submissionform"/>
    <input type="text" class="d-none" name="srctype_${uid}" value="${mimetypes[srctype]}" form="submissionform"/>
    <td><div class="table-outer-div"><div class="table-inner-div">${file.name}</div></div></td>
    <td class="centered">
        <div class="table-outer-div">
        <div class="table-inner-div">
        ${filesize}
        </div>
        </div>
    </td>
    <td>
    <span id=${file.lastModified} class="btn btn-danger delbtn"><i id=${file.lastModified} class="bi bi-trash3-fill"></i></span></td>
    `;
    content.append(parentr);
    let fi = content.querySelector(`input[name="file_${uid}"]`);
    let datr = new DataTransfer();
    datr.items.add(file);
    fi.files = datr.files;
    return content;
}
function fListRowHTMLerr(file, err_msg, cnt) {
    let content = new DocumentFragment();
    let parentr = document.createElement('tr');
    parentr.classList.add('invalid-row');
    parentr.innerHTML = `
    <td>${file.name}</td>
    <td>
        <span class="text-danger">${err_msg}</span>
    </td>
    <td><span id=${file.lastModified} class="btn btn-danger delbtn"><i id=${file.lastModified} class="bi bi-trash3-fill"></i></span></td>
    `;
    content.append(parentr);
    return content;
}
function removeExt(ext,ar){
    for(var i=0;i<ar.length;i++){
        if(ar[i]==ext){
        ar.splice(i,1);
    }}
    return ar;
}

function getTargets(mimetype) {
    if (mimetype.slice(0,5) === 'image' && mimetype !== 'image/svg+xml'){
        var ext=mimetype.slice(6).toUpperCase();
        if(ext=="JPEG") ext="JPG"
        var ar=['JPG', 'PNG', 'GIF', 'BMP', 'TIFF', 'ICO', 'ICNS', 'WEBP', 'TGA', 'PDF'];
        
        return removeExt(ext,ar);
    }

    else if (mimetype === 'application/pdf'){
        var ext=mimetype.slice(6).toUpperCase();
        if(ext=="JPEG") ext="JPG"
        var ar=['DOCX', 'JPG', 'PNG', 'GIF', 'BMP', 'TIFF', 'ICO', 'ICNS', 'WEBP', 'TGA'];
        return removeExt(ext,ar);
    }
    else if (mimetype === 'audio/mpeg') return ['WAV', 'PDF'];
    else if (mimetype === 'audio/wav') return ['MP3', 'PDF'];
    else if (mimetype === 'text/html') return ['PDF'];
    else if (mimetype === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') return ['PDF'];
    else if (mimetype === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') return ['CSV', 'TSV'];
    else return [];
}

function uuidv4() {
    // https://stackoverflow.com/a/2117523
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

function allText(string) {
    for (let i=0; i<string.length; i++) {
        let cc = string.codePointAt(i);
        if (! (cc >= 0x20 && cc < 0x7f ||
               cc >= 0x08 && cc < 0x0e ||
               cc === 0 || cc === 0x1b))
        return false;
    }
    return true;
}

function inspectFile(content, giventype) {
    if (content.slice(0,4) === '\xFF\xD8\xFF\xDB' || 
        content.slice(0,4) === '\xFF\xD8\xFF\xE0' || 
        content.slice(0,4) === '\xFF\xD8\xFF\xE1' || 
        content.slice(0,4) === '\xFF\xD8\xFF\xE2' || 
        content.slice(0,4) === '\xFF\xD8\xFF\xEE') {
        return "image/jpeg";
    } else if (content.slice(0,2) === 'BM') {
        return "image/bmp";
    } else if (content.slice(0,6) === 'GIF87a' || 
               content.slice(0,6) === 'GIF89a') {
        return "image/gif";
    } else if (content.slice(0,4) === '\x00\x00\x01\x00') {
        return "image/vnd.microsoft.icon";
    } else if (content.slice(0,4) === 'icns') {
        return "image/x-icns";
    } else if (content.slice(0,8) === '\x89\x50\x4E\x47\x0D\x0A\x1A\x0A') {
        return "image/png";
    } else if (content.slice(0,4) === '\x49\x49\x2A\x00' || 
               content.slice(0,4) === '\x4D\x4D\x00\x2A') {
        return "image/tiff";
    } else if (content.slice(0,4)  === 'RIFF' && 
               content.slice(8,12) === 'WEBP') {
        return "image/webp";
    } else if (content.slice(0,5) === '%PDF-') {
        return "application/pdf";
    } else if (content.slice(0,2) === '\xFF\xFB' ||
               content.slice(0,2) === '\xFF\xF3' ||
               content.slice(0,2) === '\xFF\xF2' ||
               content.slice(0,3) === 'ID3' ) {
        return "audio/mpeg";
    } else if (content.slice(0,4)  === 'RIFF' && 
               content.slice(8,12) === 'WAVE') {
        return "audio/wav";
    } else {
        return giventype;
    } 
}

function deleteFileList(id) {
    // let fileListArray = Array.from(finput.files);
    // let index = -1;
    // for(let i=0;i<fileListArray.length;i++){
    //     if(fileListArray[i].lastModified == parseInt(id)) {
    //         index = i;
    //         break;
    //     }
    // }
    // // Remove the file at the specified index from the array
    // fileListArray.splice(index, 1);
    // console.log(fileListArray);
    // // Create a new FileList object from the modified array
    // var newFileList = new FileList();
    // // Assign the new FileList object back to the files property of the file input element
    // Object.defineProperty(finput, "files", {
    //     value: newFileList,
    //     writable: false
    // });
    FileListLength--;
}

function fileAddedHandler(loadevent) {
    FileListLength = (Array.from(finput.files)).length;
    var rows = new DocumentFragment();
    Array.prototype.forEach.call(finput.files, function(file) {
        if (file.size > _maxFileMegaBytes*1024*1024) {
            rows.appendChild(fListRowHTMLerr(file, "File Size Too Large"));
        } else {
            var reader = new FileReader();
            reader.onloadend = () => { 
                let type = inspectFile(reader.result, file.type);
                if (getTargets(type).length === 0)
                    rows.appendChild(fListRowHTMLerr(file, "Unsupported / Corrupted Format"));
                else
                    rows.appendChild(fListRowHTML(file, type));
            }
            reader.readAsBinaryString(file.slice(0,20));
        }
    })
    window.setTimeout(function(e) {
        for (let dbn of rows.querySelectorAll('.delbtn')) {
            dbn.onclick = (event) => {
                let id = event.target.id;
                // remove the file from finput
                deleteFileList(id);
                if(FileListLength == 0) {
                    submitBtn.classList.add('d-none');
                }
                event.target.closest('tr').remove()
            }
        }
        fList.appendChild(rows);
        if (FileListLength > 0) {
            submitBtn.classList.remove('d-none');
        }
        else {
            submitBtn.classList.add('d-none');
        }
    }, 10);
}

