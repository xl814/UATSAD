import { useState } from "react";

function submitForm(ans: string){
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if(ans.toLowerCase() === 'abc'){
                resolve();
            } else {
                reject(new Error('Incorrect, try again!'));
            }
        }, 2000);
    });
}

export default function Form(){
    const [ans, setAns] = useState('');
    const [error, setError] = useState(null);
    const [status, setStatus] = useState('typing');

    if (status == 'success'){
        return (
            <div>
                <h2>Success!</h2>
                <p>You answered correctly!</p>
            </div>
        )
    }

    async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();
        setStatus('submitting');
        try{
            await submitForm(ans);
            setStatus('success');
        } catch(error: any){
            setStatus('typing');
            setError(error);
        }
    }

    const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setAns(e.target.value);
    }

    return (
        <>
             <h2>City quiz</h2>
            <p>
                In which city is there a billboard that turns air into drinkable water?
            </p>
            <form onSubmit={handleSubmit}>
                <textarea value={ans} onChange={handleChange} disabled = {status === 'submitting'}></textarea>
                <br />
                <button disabled={ans === '' || status === 'submitting'}>Submit</button>
                {error !== null && <p style={{color: 'red'}}>{error.message}</p>}
            </form>
        </>
    )
}